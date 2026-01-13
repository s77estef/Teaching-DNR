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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
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

TRAIN_SAMPLES = 150
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
REWARD_SOURCE_FILENAME = "reward_funcs.py"
CHECKPOINT_BATCH_LOG_FILENAME = "checkpoint_batch_samples.jsonl"
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
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
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
    "per_device_train_batch_size": 20,
    # gas 16: 37GB, gas 8: 38GB
    "gradient_accumulation_steps": 6,
    # with: 26GB, without: 37GB
    #"gradient_checkpointing": True,
    "num_train_epochs": 1,
    "bf16": True,
    "max_completion_length": 512,
    "num_generations": 4,
    #"do_sample": True,
    "temperature": 2, # (default 1.0) higher, 1.1 or 1.2
    "top_p": 0.7, # (default 1.0) lower, 0.9 
    #"top_k": 50, # (default None)
    "max_prompt_length": 256,
    #"report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 1,
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

FULL_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*(?P<label>harmful|unharmful)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

OPEN_THINK_RE   = re.compile(r"<\s*think\s*>", re.IGNORECASE)
CLOSE_THINK_RE  = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
OPEN_ANSWER_RE  = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
CLOSE_ANSWER_RE = re.compile(r"<\s*/\s*answer\s*>", re.IGNORECASE)

FORBIDDEN_RE = re.compile(
    r"</?\s*(reasoning)\s*>",
    re.IGNORECASE,
)

def format_reward_simple(completions, **_):
    rewards = []
    for completion in completions:
        text = completion[0].get("content") or ""

        # 0 reward if forbidden tag appears anywhere
        if FORBIDDEN_RE.search(text):
            rewards.append(0.0)
            continue

        # Count outer tags
        n_open_think   = len(OPEN_THINK_RE.findall(text))
        n_close_think  = len(CLOSE_THINK_RE.findall(text))
        n_open_answer  = len(OPEN_ANSWER_RE.findall(text))
        n_close_answer = len(CLOSE_ANSWER_RE.findall(text))

        # 0 reward if any tag is doubled or missing
        if (
            n_open_think   != 1 or
            n_close_think  != 1 or
            n_open_answer  != 1 or
            n_close_answer != 1
        ):
            rewards.append(0.0)
            continue

        score = 0.0

        # +0.1 for each exactly-once tag
        score += 0.1  # <think>
        score += 0.1  # </think>
        score += 0.1  # <answer>
        score += 0.1  # </answer>

        # +0.1 nothing before <think>
        if re.match(r"^\s*<\s*think\s*>", text, flags=re.IGNORECASE):
            score += 0.1

        # +0.1 nothing after </answer>
        if re.search(r"</\s*answer\s*>\s*$", text, flags=re.IGNORECASE):
            score += 0.1

        # +0.4 full correct pattern
        if FULL_RE.match(text):
            score += 0.4

        rewards.append(1.0 if score > 1.0 else score)

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


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _normalize_list(value: Any, batch_size: int) -> List[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return list(value)
    return [value] * batch_size


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first.get("content") or "")
    return str(completion)


def _write_reward_batch_log(step: int, output_dir: str, payload: Dict[str, Any]) -> None:
    completions = payload.get("completions") or []
    if not isinstance(completions, list):
        completions = [completions]
    rewards_by_func = payload.get("rewards") or {}
    batch_size = len(completions)
    raw_prompts = payload.get("prompts")
    raw_solutions = payload.get("solutions")
    prompts_list = list(raw_prompts) if isinstance(raw_prompts, (list, tuple)) else None
    solutions_list = list(raw_solutions) if isinstance(raw_solutions, (list, tuple)) else None
    num_prompts = len(prompts_list) if prompts_list else 0
    group_size = (
        batch_size // num_prompts
        if num_prompts and batch_size % num_prompts == 0
        else None
    )
    prompts = _normalize_list(raw_prompts, batch_size)
    solutions = _normalize_list(raw_solutions, batch_size)

    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_file = checkpoint_dir / CHECKPOINT_BATCH_LOG_FILENAME
    with output_file.open("w", encoding="utf-8") as fout:
        for idx in range(batch_size):
            prompt_idx = idx // group_size if group_size else idx
            prompt_val = (
                prompts_list[prompt_idx]
                if prompts_list and prompt_idx < len(prompts_list)
                else prompts[idx]
            )
            solution_val = (
                solutions_list[prompt_idx]
                if solutions_list and prompt_idx < len(solutions_list)
                else solutions[idx]
            )
            entry = {
                "step": step,
                "entry_type": "completion",
                "prompt": _safe_json_value(prompt_val),
                "completion": _completion_to_text(completions[idx]),
                "solution": _safe_json_value(solution_val),
            }
            fout.write(json.dumps(entry, ensure_ascii=True) + "\n")
            if group_size and (idx + 1) % group_size == 0:
                group_start = idx + 1 - group_size
                group_rewards = {}
                for name, rewards in rewards_by_func.items():
                    if isinstance(rewards, (list, tuple)) and len(rewards) >= idx + 1:
                        group_rewards[name] = list(rewards[group_start:idx + 1])
                    else:
                        group_rewards[name] = None
                summary = {
                    "step": step,
                    "entry_type": "group_rewards",
                    "prompt": _safe_json_value(prompt_val),
                    "solution": _safe_json_value(solution_val),
                    "completion_count": group_size,
                    "rewards": group_rewards,
                }
                fout.write(json.dumps(summary, ensure_ascii=True) + "\n")


class _RewardLogState:
    def __init__(self, num_funcs: int, output_dir: str, save_steps: int | None) -> None:
        self.num_funcs = num_funcs
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.current_step = 0
        self.is_main_process = True
        self.active = False
        self.logged_steps: set[int] = set()
        self.pending: Dict[int, Dict[str, Any]] = {}

    def update_step(self, step: int, is_main_process: bool) -> None:
        self.current_step = step
        self.is_main_process = is_main_process
        self.active = (
            bool(self.save_steps)
            and step > 0
            and step % int(self.save_steps) == 0
        )


def _make_logging_reward(func, log_state: _RewardLogState):
    name = getattr(func, "__name__", "reward_func")

    def wrapped(completions, **kwargs):
        rewards = func(completions, **kwargs)
        if not (log_state.active and log_state.is_main_process):
            return rewards
        step = log_state.current_step
        if step in log_state.logged_steps:
            return rewards
        payload = log_state.pending.setdefault(
            step,
            {"rewards": {}, "completions": completions, "prompts": None, "solutions": None},
        )
        if payload["prompts"] is None:
            payload["prompts"] = kwargs.get("prompts") or kwargs.get("prompt")
        if payload["solutions"] is None:
            payload["solutions"] = kwargs.get("solutions") or kwargs.get("solution")
        payload["rewards"][name] = rewards
        if len(payload["rewards"]) >= log_state.num_funcs:
            _write_reward_batch_log(step, log_state.output_dir, payload)
            log_state.logged_steps.add(step)
            log_state.pending.pop(step, None)
        return rewards

    return wrapped


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
    _write_reward_source(training_args.output_dir)

    reward_funcs = [format_reward_simple]
    save_steps = training_args.save_steps or GRPO_TRAINING_CONFIG.get("save_steps")
    log_state = _RewardLogState(
        num_funcs=len(reward_funcs),
        output_dir=training_args.output_dir,
        save_steps=save_steps,
    )
    logged_reward_funcs = [
        _make_logging_reward(func, log_state) for func in reward_funcs
    ]

    class _RewardBatchLogger(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            step = state.global_step + 1
            is_main = getattr(state, "is_world_process_zero", True)
            log_state.update_step(step, is_main)
            return control

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=logged_reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.add_callback(_RewardBatchLogger())
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
