# new sys prompt: import, NORMATIVE=True/False, SYSTEM_PROMPT=, prompt length, adjust check.py


#!/usr/bin/env python
import json
import os
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

from fine_tune.shared import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FS,
    SYSTEM_PROMPT_NORMATIVE,
    SYSTEM_PROMPT_NNORMATIVE,
    SYSTEM_PROMPT_TNORMATIVE,
    SYSTEM_PROMPT_GPT3,
    SYSTEM_PROMPT_NEWNNORMATIVE,
    extract_label_if_any,
    load_wildguard_train_rendered,
    hf_cli_login,
    wandb_cli_login,
)
from fine_tune.train_logger import RewardLogger, completion_to_text

# ---- config settings ----

TRAIN_SAMPLES = 10000
#MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
#MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
#MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen3-4B"
#MODEL_ID = "Qwen/Qwen3.5-4B"

DEBUG = False
NORMATIVE = True # changes system prompt for normative reasoning

if NORMATIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_GPT3


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = MODEL_ID.split("/", 1)[-1]
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "trained_experiments",
    f"{model_name}-GRPO-test_{timestamp}",
)
GRPO_CONFIG_FILENAME = "grpo_config.json"
REWARD_SOURCE_FILENAME = "reward_funcs.py"
GRPO_TRAINING_CONFIG_C = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-5,
    "beta": 0.025,
    "remove_unused_columns": False,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 4,
    # model-specific defaults: {'temperature': 0.6, 'top_p': 0.95}
    #"temperature": 1.1, # 2.0 is way too much
    #"top_p": 0.9, # keep smallest set of tokens whose cumulative probability ≥ p
    #"top_k": 50, # (default None) keep k most probable tokens
    #"min_p": 0.05, # discard any token whose individual probability < p
    #"repetition_penalty": 1.05, # penalize repeated tags or loops
    #"generation_kwargs": {
    #    "do_sample": True,
    #},
    # 713 for gpt3?, 487 for gpt2, 442 for gpt, 370 for tnormative, 337 for one of the normative, 256 for normal system prompt
    "max_prompt_length": 576, # 256 + 81 because the normative system prompt is 81 tokens longer
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}
GRPO_TRAINING_CONFIG_B = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-5,
    "beta": 0.025,
    "remove_unused_columns": False,
    # pdtbs 8: 37GB, pdtbs 16: 41-OOM
    # good: pdtbs 21, gas none, 43GB
    "per_device_train_batch_size": 5,
    # gas 16: 37GB, gas 8: 38GB
    "gradient_accumulation_steps": 16,
    # with: 26GB, without: 37GB
    #"gradient_checkpointing": True,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 4,
    #"temperature": 0.7, # (default 1.0) 2.0 is way too much
    #"top_p": 0.8, # (default 1.0) keep smallest set of tokens whose cumulative probability ≥ p
    #"top_k": 50, # (default None) keep k most probable tokens
    #"min_p": 0.05, # discard any token whose individual probability < p
    #"repetition_penalty": 1.05, # penalize repeated tags or loops
    #"generation_kwargs": {
    #    "do_sample": True,
    #},
    "max_prompt_length": 1055,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 50,
}
GRPO_TRAINING_CONFIG = GRPO_TRAINING_CONFIG_B

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


# ------------------------------------

def load_lora_model(lora_overrides: Dict[str, Any] | None = None) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
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

# --------------------------------------------------------- REWARDS

 

"""

def format_reward(completions, **_):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        rewards.append(1.0 if FULL_RE.match(text) else 0.0)
    return rewards

"""

def accuracy_reward(completions, **kwargs):
    # Strict accuracy reward: FULL_RE must match exactly, label must be extractable, label must equal gold
    solutions = kwargs["solution"]
    rewards = []

    for completion, solution in zip(completions, solutions):
        if DEBUG:
            print("COMPLETION:", completion)
        text = completion_to_text(completion).strip()
        if DEBUG:
            print("TEXT:", text)
        pred = extract_label_if_any(text, include_normative_reasoning=NORMATIVE)

        if pred is None:
            rewards.append(0.0)
            continue

        gold = (solution or "").lower()
        rewards.append(1.0 if pred == gold else 0.0)

    return rewards


# --------------------------------------------------------- REWARDS



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


def run_trainer(
    model: torch.nn.Module,
    dataset: Dataset,
    training_args: GRPOConfig,
    training_config: Dict[str, Any] | None = None,
) -> GRPOTrainer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or GRPO_TRAINING_CONFIG)
    _write_grpo_config(config_for_log, training_args.output_dir)
    _write_reward_source(training_args.output_dir)

    reward_funcs = [accuracy_reward]
    save_steps = training_args.save_steps or GRPO_TRAINING_CONFIG.get("save_steps")
    reward_logger = RewardLogger(
        model_id=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        output_dir=training_args.output_dir,
        save_steps=save_steps,
        num_funcs=len(reward_funcs),
    )
    logged_reward_funcs = [
        reward_logger.wrap_reward(func) for func in reward_funcs
    ]

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
    trainer.train(resume_from_checkpoint="fine_tune/trained_experiments/Qwen3-4B-GRPO-rubric_plus_accuracy_20260422_011719/checkpoint-500")
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

    train_ds = load_wildguard_train_rendered(
        num_samples=TRAIN_SAMPLES,
        max_prompt_tokens=GRPO_TRAINING_CONFIG["max_prompt_length"],
        tokenizer_name=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        only_adversarial=True
    )
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, GRPO_TRAINING_CONFIG)



if __name__ == "__main__":
    main()
