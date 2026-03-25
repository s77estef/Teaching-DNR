#!/usr/bin/env python
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from fine_tune.reward_funcs_judge import (
    judge_plus_accuracy_reward,
    judge_reward,
    judge_with_gold_direction_reward,
)
from fine_tune.shared import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_GPT3,
    hf_cli_login,
    load_wildguard_train_rendered,
    validate_and_extract_label,
    wandb_cli_login,
)
from fine_tune.train_logger import RewardLogger, completion_to_text

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.bfloat16)

TRAIN_SAMPLES = 10000
MODEL_ID = "Qwen/Qwen3-4B"
#MODEL_ID = "Qwen/Qwen3.5-4B"
NORMATIVE = True

if NORMATIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_GPT3

REWARD_MODE = "rubric_with_gold_direction"
GRPO_CONFIG_FILENAME = "grpo_config.json"
REWARD_SOURCE_FILENAME = "reward_funcs.py"

JUDGE_GRPO_TRAINING_CONFIG: Dict[str, object] = {
    "learning_rate": 1e-5,
    "beta": 0.025,
    "remove_unused_columns": False,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 2,
    "max_prompt_length": 1055, # 576(+12) for sys, 1043(+12) for gpt3
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}

LORA_CONFIG: Dict[str, Any] = {
    "task_type": "CAUSAL_LM",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def load_lora_model(lora_overrides: Dict[str, Any] | None = None) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto" if device == "cuda" else None,
    )
    lora_params = copy.deepcopy(LORA_CONFIG)
    if lora_overrides:
        lora_params.update(lora_overrides)
    lora_cfg = LoraConfig(**lora_params)
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _write_grpo_config(
    config: Dict[str, Any],
    output_dir: str,
    results: Dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"config": config}
    if results:
        payload["results"] = results
    config_path = output_path / GRPO_CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)
    return config_path


def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _write_reward_source(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_text(encoding="utf-8")
    marker = "# --------------------------------------------------------- REWARDS"
    start = source.find(marker)
    reward_block = source
    if start != -1:
        end = source.find(marker, start + len(marker))
        if end != -1:
            end_line = source.find("\n", end)
            if end_line == -1:
                end_line = len(source)
            reward_block = source[start:end_line].rstrip() + "\n"
        else:
            reward_block = source[start:]
    output_file = output_path / REWARD_SOURCE_FILENAME
    output_file.write_text(reward_block, encoding="utf-8")
    return output_file


def format_gate_reward(completions, **_) -> List[float]:
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        format_ok, _ = validate_and_extract_label(
            text,
            include_normative_reasoning=NORMATIVE,
        )
        rewards.append(1.0 if format_ok else 0.0)
    return rewards


def build_training_args() -> GRPOConfig:
    config = copy.deepcopy(JUDGE_GRPO_TRAINING_CONFIG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODEL_ID.split("/", 1)[-1]
    config["output_dir"] = str(
        Path(__file__).resolve().parent
        / "trained_experiments"
        / f"{model_name}-GRPO-{REWARD_MODE}_{timestamp}"
    )
    return GRPOConfig(**config)


def _resolve_reward_funcs() -> List[Callable]:
    if REWARD_MODE == "rubric_only":
        return [format_gate_reward, judge_reward]
    if REWARD_MODE == "rubric_plus_accuracy":
        return [format_gate_reward, judge_plus_accuracy_reward]
    if REWARD_MODE == "rubric_with_gold_direction":
        return [format_gate_reward, judge_with_gold_direction_reward]
    raise ValueError(f"Unsupported reward mode: {REWARD_MODE}")


# --------------------------------------------------------- REWARDS


def run_trainer(
    model: torch.nn.Module,
    dataset: Dataset,
    training_args: GRPOConfig,
    reward_funcs: List[Callable],
    training_config: Dict | None = None,
) -> GRPOTrainer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or JUDGE_GRPO_TRAINING_CONFIG)
    config_for_log["output_dir"] = training_args.output_dir
    config_for_log["reward_mode"] = REWARD_MODE
    config_for_log["reward_funcs"] = [
        getattr(func, "__name__", "reward_func") for func in reward_funcs
    ]
    _write_grpo_config(config_for_log, training_args.output_dir)
    _write_reward_source(training_args.output_dir)

    save_steps = training_args.save_steps or config_for_log.get("save_steps")
    reward_logger = RewardLogger(
        model_id=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        output_dir=training_args.output_dir,
        save_steps=save_steps,
        num_funcs=len(reward_funcs),
    )
    logged_reward_funcs = [reward_logger.wrap_reward(func) for func in reward_funcs]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=logged_reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.add_callback(reward_logger.get_callback())
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    trainer.save_model(training_args.output_dir)
    final_results = {
        "final_steps": trainer.state.global_step,
        "training_time": _format_duration(elapsed),
    }
    _write_grpo_config(config_for_log, training_args.output_dir, final_results)
    return trainer


# --------------------------------------------------------- REWARDS


def main() -> None:
    hf_cli_login()
    wandb_cli_login()

    train_ds = load_wildguard_train_rendered(
        num_samples=TRAIN_SAMPLES,
        max_prompt_tokens=JUDGE_GRPO_TRAINING_CONFIG["max_prompt_length"],
        tokenizer_name=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        only_adversarial=True,
    )
    print(train_ds)

    reward_funcs = _resolve_reward_funcs()
    model = load_lora_model()
    training_args = build_training_args()
    run_trainer(
        model=model,
        dataset=train_ds,
        training_args=training_args,
        reward_funcs=reward_funcs,
        training_config=JUDGE_GRPO_TRAINING_CONFIG,
    )


if __name__ == "__main__":
    main()
