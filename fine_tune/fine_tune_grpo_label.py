#!/usr/bin/env python
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence, Tuple
import copy
import math


import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.bfloat16) # not sure if necessary
# TODO batch generation and batch tokenization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fine_tune.shared import SYSTEM_PROMPT, SYSTEM_PROMPT_FS, load_wildguard_train, hf_cli_login, wandb_cli_login

# ---- config settings ----

TRAIN_SAMPLES = 100
#MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
#MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
#MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen3-4B"
DEBUG = False

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = MODEL_ID.split("/", 1)[-1]
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "trained_experiments",
    f"{model_name}-GRPO-test_{timestamp}",
)
GRPO_CONFIG_FILENAME = "grpo_config.json"
GRPO_TRAINING_CONFIG_C = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-5,
    "remove_unused_columns": False,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 256,
    "num_generations": 4,
    "max_prompt_length": 256,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}
GRPO_TRAINING_CONFIG_B = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-4,
    "remove_unused_columns": False,
    # pdtbs 8: 37GB, pdtbs 16: 41-OOM
    # good: pdtbs 21, gas none, 43GB
    "per_device_train_batch_size": 21,
    # gas 16: 37GB, gas 8: 38GB
    #"gradient_accumulation_steps": 4,
    # with: 26GB, without: 37GB
    #"gradient_checkpointing": True,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 3,
    "max_prompt_length": 256,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}
GRPO_TRAINING_CONFIG = GRPO_TRAINING_CONFIG_C

LORA_CONFIG: Dict[str, Any] = {
    "task_type": "CAUSAL_LM",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,  # possibly higher dropout with 0.1?
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


# ------------------------------------


def attach_prompts(dataset: Dataset) -> Dataset:
    def make_conversation(example: Dict) -> Dict[str, List[Dict[str, str]]]:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ]
        }
    dataset = dataset.rename_column("prompt_harm_label", "solution")
    dataset = dataset.map(lambda ex: {**ex, "solution": ex["solution"].lower()})
    return dataset.map(make_conversation)


def load_lora_model(lora_overrides: Dict[str, Any] | None = None) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto" if device == "cuda" else None,
    )
    lora_params = copy.deepcopy(LORA_CONFIG)
    if lora_overrides:
        lora_params.update(lora_overrides)
    lora_cfg = LoraConfig(**lora_params)
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model




FORMAT_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>\s*(?P<label>harmful|unharmful)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
FORBIDDEN_IN_THINK_RE = re.compile(
    r"</?\s*(answer|think|reasoning)\s*>",
    re.IGNORECASE,
)

def validate_and_extract_label(text: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (format_ok, label) where label is normalized to 'harmful'/'unharmful' if present.
    """
    m = FORMAT_RE.match(text or "")
    if not m:
        return False, None

    think_text = m.group("think") or ""
    if FORBIDDEN_IN_THINK_RE.search(think_text):
        return False, None

    label = (m.group("label") or "").lower()
    return True, label

def format_reward(completions, **_):
    rewards = []
    for completion in completions:
        text = completion[0].get("content") or ""
        ok, _ = validate_and_extract_label(text)
        rewards.append(1.0 if ok else 0.0)
    return rewards

def accuracy_reward(completions: Sequence[Sequence[Dict[str, str]]], **kwargs) -> List[float]:
    solutions = kwargs["solution"]
    rewards: List[float] = []

    for completion, solution in zip(completions, solutions):
        text = completion[0].get("content") or ""
        ok, pred = validate_and_extract_label(text)
        if not ok or pred is None:
            rewards.append(0.0)
            continue

        gold = (solution or "").lower()
        rewards.append(1.0 if pred == gold else 0.0)

    return rewards


def build_training_args() -> GRPOConfig:
    return GRPOConfig(**GRPO_TRAINING_CONFIG)

# only for training documentation
def _write_grpo_config(config: Dict[str, Any], output_dir: str, results: Dict[str, Any] | None = None) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"config": config}
    if results:
        payload["results"] = results
    config_path = output_path / GRPO_CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)
    return config_path

# only for training documentation
def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def run_trainer(
    model: torch.nn.Module,
    dataset: Dataset,
    training_args: GRPOConfig,
    training_config: Dict[str, Any] | None = None,
) -> GRPOTrainer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or GRPO_TRAINING_CONFIG)
    _write_grpo_config(config_for_log, training_args.output_dir)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=dataset,
    )
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    #trainer.train(resume_from_checkpoint="fine_tune/trained_experiments/name_here/checkpoint-1000")
    trainer.save_model(training_args.output_dir)
    final_results = {
        "final_steps": trainer.state.global_step,
        "training_time": _format_duration(elapsed),
    }
    _write_grpo_config(config_for_log, training_args.output_dir, final_results)
    return trainer


def main() -> None:
    hf_cli_login()
    wandb_cli_login()
    train_ds = load_wildguard_train(num_samples=TRAIN_SAMPLES, max_tokens=GRPO_TRAINING_CONFIG["max_completion_length"]) # 170 = 256 (max_prompt_length) - 86 (system prompt)
    train_ds = attach_prompts(train_ds)
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, GRPO_TRAINING_CONFIG)


if __name__ == "__main__":
    main()
