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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
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
    extract_label_if_any,
    load_wildguard_train_rendered,
    hf_cli_login,
    wandb_cli_login,
)

# ---- config settings ----

TRAIN_SAMPLES = 10000
#MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
#MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
#MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen3-4B"
DEBUG = False
NORMATIVE = True # changes system prompt for normative reasoning

if NORMATIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_NORMATIVE

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
    "max_prompt_length": 337, # 256 + 81 because the normative system prompt is 81 tokens longer
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}
GRPO_TRAINING_CONFIG_B = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-5,
    #"beta": 0.02, # KL regularization, keeps policy close to reference
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
    "temperature": 0.7, # (default 1.0) 2.0 is way too much
    #"top_p": 0.8, # (default 1.0) keep smallest set of tokens whose cumulative probability ≥ p
    #"top_k": 50, # (default None) keep k most probable tokens
    #"min_p": 0.05, # discard any token whose individual probability < p
    #"repetition_penalty": 1.05, # penalize repeated tags or loops
    #"generation_kwargs": {
    #    "do_sample": True,
    #},
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

def completion_to_text(completion) -> str:
    # TRL common shape: [{"content": "..."}]
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"] or "")
        if isinstance(first, str):
            return first
    # fallback
    if isinstance(completion, str):
        return completion
    return ""

"""

def format_reward(completions, **_):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        rewards.append(1.0 if FULL_RE.match(text) else 0.0)
    return rewards

"""

# TODO: delete this if i only use think
REASONING_TAG = "think"

def accuracy_reward(completions, **kwargs):
    # Strict accuracy reward: FULL_RE must match exactly, label must be extractable, label must equal gold
    solutions = kwargs["solution"]
    rewards = []

    for completion, solution in zip(completions, solutions):
        text = completion_to_text(completion).strip()
        pred = extract_label_if_any(text, reasoning_tag=REASONING_TAG)

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


def _extract_system_user(prompt_val: Any) -> Tuple[str | None, str | None]:
    if isinstance(prompt_val, list):
        system = None
        user = None
        for msg in prompt_val:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system" and system is None:
                system = msg.get("content")
            elif role == "user" and user is None:
                user = msg.get("content")
        return system, user
    if isinstance(prompt_val, dict):
        return None, prompt_val.get("content")
    return None, str(prompt_val)


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
        def _write_entry(entry: Dict[str, Any]) -> None:
            fout.write(json.dumps(entry, ensure_ascii=True, indent=2))
            fout.write("\n\n")

        system_prompt = SYSTEM_PROMPT
        if prompts:
            extracted_system, _ = _extract_system_user(prompts[0])
            if extracted_system:
                system_prompt = extracted_system
        header = {"step": step, "system_prompt": system_prompt}
        _write_entry(header)

        if batch_size == 0:
            return

        if group_size:
            prompt_keys = []
            for prompt_val in prompts_list or []:
                _, user_prompt = _extract_system_user(prompt_val)
                prompt_keys.extend([user_prompt] * group_size)
        else:
            prompt_keys = []
            for prompt_val in prompts:
                _, user_prompt = _extract_system_user(prompt_val)
                prompt_keys.append(user_prompt)

        groups: List[Tuple[int, int]] = []
        start_idx = 0
        for idx in range(1, batch_size):
            if prompt_keys[idx] != prompt_keys[idx - 1]:
                groups.append((start_idx, idx))
                start_idx = idx
        groups.append((start_idx, batch_size))

        for start_idx, end_idx in groups:
            prompt_val = prompts[start_idx]
            solution_val = solutions[start_idx]
            _, user_prompt = _extract_system_user(prompt_val)
            completion_group = [
                completion_to_text(item) for item in completions[start_idx:end_idx]
            ]
            reward_totals = []
            for local_idx in range(start_idx, end_idx):
                total = 0.0
                for rewards in rewards_by_func.values():
                    if isinstance(rewards, (list, tuple)) and local_idx < len(rewards):
                        try:
                            total += float(rewards[local_idx])
                        except (TypeError, ValueError):
                            total += 0.0
                reward_totals.append(total)
            entry = {
                "prompt": _safe_json_value(user_prompt),
                "solution": _safe_json_value(solution_val),
                "completions": completion_group,
                "rewards": reward_totals,
            }
            _write_entry(entry)


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or GRPO_TRAINING_CONFIG)
    _write_grpo_config(config_for_log, training_args.output_dir)
    _write_reward_source(training_args.output_dir)

    reward_funcs = [accuracy_reward]
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

    train_ds = load_wildguard_train_rendered(
        num_samples=TRAIN_SAMPLES,
        max_prompt_tokens=GRPO_TRAINING_CONFIG["max_prompt_length"],
        tokenizer_name=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
    )
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, GRPO_TRAINING_CONFIG)



if __name__ == "__main__":
    main()
