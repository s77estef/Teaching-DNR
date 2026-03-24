#!/usr/bin/env python
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import fine_tune.fine_tune_grpo_label as base
from fine_tune.reward_funcs_judge import judge_plus_accuracy_reward, judge_reward
from fine_tune.shared import validate_and_extract_label
from fine_tune.train_logger import RewardLogger, completion_to_text

REWARD_MODE = "rubric_only"


def format_gate_reward(completions, **_) -> List[float]:
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        format_ok, _ = validate_and_extract_label(
            text,
            include_normative_reasoning=base.NORMATIVE,
        )
        rewards.append(1.0 if format_ok else 0.0)
    return rewards


def build_training_args() -> GRPOConfig:
    training_config = copy.deepcopy(base.GRPO_TRAINING_CONFIG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = base.MODEL_ID.split("/", 1)[-1]
    training_config["output_dir"] = str(
        Path(__file__).resolve().parent
        / "trained_experiments"
        / f"{model_name}-GRPO-{REWARD_MODE}_{timestamp}"
    )
    return GRPOConfig(**training_config)


def _resolve_reward_funcs() -> List[Callable]:
    if REWARD_MODE == "rubric_only":
        return [format_gate_reward, judge_reward]
    if REWARD_MODE == "rubric_plus_accuracy":
        return [format_gate_reward, judge_plus_accuracy_reward]
    raise ValueError(f"Unsupported reward mode: {REWARD_MODE}")


# --------------------------------------------------------- REWARDS


def run_trainer(
    model: torch.nn.Module,
    dataset: Dataset,
    training_args: GRPOConfig,
    reward_funcs: List[Callable],
    training_config: Dict | None = None,
) -> GRPOTrainer:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or base.GRPO_TRAINING_CONFIG)
    config_for_log["output_dir"] = training_args.output_dir
    config_for_log["reward_mode"] = REWARD_MODE
    config_for_log["reward_funcs"] = [
        getattr(func, "__name__", "reward_func") for func in reward_funcs
    ]
    base._write_grpo_config(config_for_log, training_args.output_dir)
    base._write_reward_source(training_args.output_dir)

    save_steps = training_args.save_steps or base.GRPO_TRAINING_CONFIG.get("save_steps")
    reward_logger = RewardLogger(
        model_id=base.MODEL_ID,
        system_prompt=base.SYSTEM_PROMPT,
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
        "training_time": base._format_duration(elapsed),
    }
    base._write_grpo_config(config_for_log, training_args.output_dir, final_results)
    return trainer


# --------------------------------------------------------- REWARDS


def main() -> None:
    base.hf_cli_login()
    base.wandb_cli_login()

    train_ds = base.load_wildguard_train_rendered(
        num_samples=base.TRAIN_SAMPLES,
        max_prompt_tokens=base.GRPO_TRAINING_CONFIG["max_prompt_length"],
        tokenizer_name=base.MODEL_ID,
        system_prompt=base.SYSTEM_PROMPT,
        only_adversarial=True,
    )
    print(train_ds)

    reward_funcs = _resolve_reward_funcs()
    model = base.load_lora_model()
    training_args = build_training_args()
    run_trainer(
        model=model,
        dataset=train_ds,
        training_args=training_args,
        reward_funcs=reward_funcs,
        training_config=copy.deepcopy(base.GRPO_TRAINING_CONFIG),
    )


if __name__ == "__main__":
    main()
