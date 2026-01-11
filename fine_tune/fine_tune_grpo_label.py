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

TRAIN_SAMPLES = 10000
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
    "learning_rate": 1e-5,
    "remove_unused_columns": False,
    # pdtbs 8: 37GB, pdtbs 16: 41-OOM
    # good: pdtbs 21, gas none, 43GB
    "per_device_train_batch_size": 20,
    # gas 16: 37GB, gas 8: 38GB
    "gradient_accumulation_steps": 6,
    # with: 26GB, without: 37GB
    #"gradient_checkpointing": True,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 4,
    "max_prompt_length": 256,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}
GRPO_TRAINING_CONFIG = GRPO_TRAINING_CONFIG_B

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

# --------------------------------------------------------- REWARDS


# --- Core patterns ---
FULL_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>\s*(?P<label>harmful|unharmful)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

# skeleton match: exactly one think-block, then whitespace-only gap, then one answer-block, nothing else (except surrounding whitespace)
# label correctness is handled separately
SKELETON_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>(?P<gap>\s*)<answer>(?P<ans>.*?)</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

# forbidden tags inside <think> (and usually anywhere extra)
FORBIDDEN_TAGS_RE = re.compile(
    r"</?\s*(answer|think|reasoning)\s*>",
    re.IGNORECASE,
)

# Count occurrences of the outer tags (case-insensitive, flexible whitespace)
OPEN_THINK_RE  = re.compile(r"<\s*think\s*>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
OPEN_ANSWER_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
CLOSE_ANSWER_RE= re.compile(r"<\s*/\s*answer\s*>", re.IGNORECASE)

# For ends with </answer> except whitespace
ENDS_WITH_CLOSE_ANSWER_RE = re.compile(r"</\s*answer\s*>\s*$", re.IGNORECASE)


def _count(pat: re.Pattern, s: str) -> int:
    return len(pat.findall(s or ""))


def format_score(text: str) -> float:
    """
    Returns a shaped format score in [0.0, 1.0].

    Design goals:
    - Exact FULL_RE match gets 1.0.
    - Near-miss structure gets partial credit but is capped (<= 0.7).
    - Duplicates/extra tags and forbidden tags are penalized enough that malformed outputs
      don't score high.
    - Strong preference for: starts with <think>, ends with </answer>, whitespace-only gap.
    """
    text = text or ""

    # Exact match: immediately return 1.0
    if FULL_RE.match(text):
        think = FULL_RE.match(text).group("think") or ""
        if FORBIDDEN_TAGS_RE.search(think):
            # disallow any tag strings inside think, even in exact matches
            return 0.0
        return 1.0

    # Shaped scoring for non-exact outputs (capped later)
    score = 0.0

    # Counts of outer tags
    n_open_think = _count(OPEN_THINK_RE, text)
    n_close_think = _count(CLOSE_THINK_RE, text)
    n_open_answer = _count(OPEN_ANSWER_RE, text)
    n_close_answer = _count(CLOSE_ANSWER_RE, text)

    # Duplicate/extra outer tags penalty:
    # - Expect exactly 1 of each outer tag.
    # - Penalize each extra/missing occurrence strongly.
    expected = {
        "open_think": (n_open_think, 1),
        "close_think": (n_close_think, 1),
        "open_answer": (n_open_answer, 1),
        "close_answer": (n_close_answer, 1),
    }
    for _, (count, exp) in expected.items():
        if count != exp:
            # Penalize deviation, extra or missing
            # example: 2 instead of 1 => -0.35, 0 instead of 1 => -0.35, 3 => -0.70
            score -= 0.35 * abs(count - exp)

    # If the skeleton matches, can award structured partial credit safely.
    m = SKELETON_RE.match(text)
    if m:
        think = m.group("think") or ""
        gap = m.group("gap") or ""
        ans = m.group("ans") or ""

        # +0.2 for having a well-ordered think block (by virtue of skeleton match)
        score += 0.2
        # +0.2 for having a well-ordered answer block (by virtue of skeleton match)
        score += 0.2

        # +0.2 if it (ignoring leading whitespace) starts with <think>
        if re.match(r"^\s*<\s*think\s*>", text, flags=re.IGNORECASE):
            score += 0.2

        # +0.2 if it ends with </answer> (except whitespace)
        if ENDS_WITH_CLOSE_ANSWER_RE.search(text):
            score += 0.2

        # +0.1 if no text between </think> and <answer> except whitespace
        if re.fullmatch(r"\s*", gap):
            score += 0.1

        # Penalize forbidden tags inside <think>
        forbidden_in_think = FORBIDDEN_TAGS_RE.findall(think)
        if forbidden_in_think:
            score -= 0.30 * len(forbidden_in_think)

        # discourage tags inside answer content too
        forbidden_in_answer = FORBIDDEN_TAGS_RE.findall(ans)
        if forbidden_in_answer:
            score -= 0.20 * len(forbidden_in_answer)

    else:
        # If skeleton doesn't match, do not award contains points, only penalties apply
        # prevents rewarding out-of-order or multi-block structures
        pass

    # Cap non-exact rewards so exact is clearly best.
    # Anything non-exact should not exceed 0.7
    if score > 0.7:
        score = 0.7

    # Clamp to [0, 1]
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return float(score)


# --- TRL reward function wrappers ---

def format_reward_shaped(completions, **_):
    #TRL expects a list of floats, one per sample
    rewards = []
    for completion in completions:
        text = completion[0].get("content") or ""
        rewards.append(format_score(text))
    return rewards

"""

def _extract_label_if_any(text: str) -> Optional[str]:
    # Extract label only if FULL_RE matches exactly. Returns normalized label or None.
    m = FULL_RE.match(text or "")
    if not m:
        return None
    return (m.group("label") or "").lower()

def accuracy_reward_strict(completions, **kwargs):
    # Strict accuracy: only award accuracy if FULL_RE matches exactly.
    # Returns 1.0 if predicted label equals gold, else 0.0.
    solutions = kwargs["solution"]
    rewards = []
    for completion, solution in zip(completions, solutions):
        text = completion[0].get("content") or ""
        pred = _extract_label_if_any(text)
        if pred is None:
            rewards.append(0.0)
            continue
        gold = (solution or "").lower()
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards
"""

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
        reward_funcs=[format_reward_shaped],
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
