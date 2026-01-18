#!/usr/bin/env python
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fine_tune.shared import SYSTEM_PROMPT, load_wildguard_train, hf_cli_login

# ---- config settings ----

NUM_SAMPLES = 30
MODEL_ID = "Qwen/Qwen3-4B"

GENERATION_CONFIG: Dict[str, Any] = {
    "max_prompt_length": 256,
    "max_completion_length": 512,
    "num_generations": 4,
    "temperature": [0.7],
    #"top_p": [0.6],
    ##"top_k": [100],
    #"min_p": [0.1],
    #"repetition_penalty": 1.05,
    #"do_sample": True,
    "batch_size": 4,
}

CHECKPOINT_BATCH_LOG_FILENAME = "checkpoint_batch_samples.jsonl"


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = MODEL_ID.split("/", 1)[-1]
OUTPUT_DIR_BASE = os.path.join(
    os.path.dirname(__file__),
    f"{model_name}-GEN-test_{timestamp}",
)

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


def load_model() -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model


# --------------------------------------------------------- REWARDS


FULL_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*(?P<label>harmful|unharmful)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Strict tags: no whitespace inside angle brackets
OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
OPEN_ANSWER_RE = re.compile(r"<answer>", re.IGNORECASE)
CLOSE_ANSWER_RE = re.compile(r"</answer>", re.IGNORECASE)

# (Optional but recommended) forbid any tag except exact think/answer open/close
FORBIDDEN_RE = re.compile(
    r"<(?!/?(?:think|answer)>)[^>]+>",
    re.IGNORECASE,
)


def format_reward(completions, **_):
    rewards = []
    for completion in completions:
        text = completion[0].get("content") or ""

        # 0 reward if forbidden tag appears anywhere
        if FORBIDDEN_RE.search(text):
            rewards.append(0.0)
            continue

        # Count tags
        n_open_think = len(OPEN_THINK_RE.findall(text))
        n_close_think = len(CLOSE_THINK_RE.findall(text))
        n_open_answer = len(OPEN_ANSWER_RE.findall(text))
        n_close_answer = len(CLOSE_ANSWER_RE.findall(text))

        score = 0.0

        # +0.1 for each tag that appears exactly once
        if n_open_think == 1:
            score += 0.1  # <think>
        if n_close_think == 1:
            score += 0.1  # </think>
        if n_open_answer == 1:
            score += 0.1  # <answer>
        if n_close_answer == 1:
            score += 0.1  # </answer>

        # +0.1 nothing before <think>
        if re.match(r"^\s*<think>", text, flags=re.IGNORECASE):
            score += 0.1

        # +0.1 nothing after </answer>
        if re.search(r"</answer>\s*$", text, flags=re.IGNORECASE):
            score += 0.1

        # +0.4 full correct pattern
        if FULL_RE.match(text):
            score += 0.4

        rewards.append(1.0 if score > 1.0 else score)

    return rewards


# --------------------------------------------------------- LOGGING


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


def _write_reward_batch_log(output_dir: str, output_filename: str, payload: Dict[str, Any]) -> None:
    completions = payload.get("completions") or []
    if not isinstance(completions, list):
        completions = [completions]
    rewards_by_func = payload.get("rewards") or {}
    batch_size = len(completions)
    raw_prompts = payload.get("prompts")
    raw_solutions = payload.get("solutions")
    prompts_list = list(raw_prompts) if raw_prompts is not None else None
    solutions_list = list(raw_solutions) if raw_solutions is not None else None
    num_prompts = len(prompts_list) if prompts_list else 0
    group_size = (
        batch_size // num_prompts
        if num_prompts and batch_size % num_prompts == 0
        else None
    )
    prompts = _normalize_list(raw_prompts, batch_size)
    solutions = _normalize_list(raw_solutions, batch_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / output_filename
    with output_file.open("w", encoding="utf-8") as fout:
        def _write_entry(entry: Dict[str, Any]) -> None:
            fout.write(json.dumps(entry, ensure_ascii=True, indent=2))
            fout.write("\n\n")

        system_prompt = SYSTEM_PROMPT
        if prompts:
            extracted_system, _ = _extract_system_user(prompts[0])
            if extracted_system:
                system_prompt = extracted_system
        header = {
            "generation_config": payload.get("generation_config"),
            "reward_mad": payload.get("reward_mad"),
            "system_prompt": system_prompt,
        }
        _write_entry(header)

        if batch_size == 0:
            return

        if group_size:
            groups: List[Tuple[int, int]] = []
            for idx in range(num_prompts):
                start_idx = idx * group_size
                end_idx = start_idx + group_size
                groups.append((start_idx, end_idx))
        else:
            prompt_keys = []
            for prompt_val in prompts:
                _, user_prompt = _extract_system_user(prompt_val)
                if user_prompt is None:
                    user_prompt = str(prompt_val)
                prompt_keys.append(user_prompt)

            groups = []
            start_idx = 0
            for idx in range(1, batch_size):
                if prompt_keys[idx] != prompt_keys[idx - 1]:
                    groups.append((start_idx, idx))
                    start_idx = idx
            groups.append((start_idx, batch_size))

        for group_idx, (start_idx, end_idx) in enumerate(groups):
            if group_size and prompts_list:
                prompt_val = prompts_list[group_idx]
            else:
                prompt_val = prompts[start_idx]
            if group_size and solutions_list:
                solution_val = solutions_list[group_idx]
            else:
                solution_val = solutions[start_idx]
            _, user_prompt = _extract_system_user(prompt_val)
            if user_prompt is None:
                user_prompt = str(prompt_val)
            completion_group = [
                _completion_to_text(item) for item in completions[start_idx:end_idx]
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


# --------------------------------------------------------- GENERATION


def _prompt_to_text(tokenizer: AutoTokenizer, prompt_val: Any) -> str:
    if isinstance(prompt_val, list):
        return tokenizer.apply_chat_template(
            prompt_val,
            tokenize=False,
            add_generation_prompt=True,
        )
    if isinstance(prompt_val, dict):
        return str(prompt_val.get("content") or "")
    return str(prompt_val)


def _batched(items: Sequence[Any], batch_size: int) -> Sequence[Sequence[Any]]:
    if batch_size <= 0:
        yield items
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _mean_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_val = sum(values) / float(len(values))
    return sum(abs(v - mean_val) for v in values) / float(len(values))


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_param(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def generate_completions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[Any],
    generation_config: Dict[str, Any],
) -> List[List[Dict[str, str]]]:
    completions: List[List[Dict[str, str]]] = []
    batch_size = int(generation_config.get("batch_size", 1))
    num_generations = int(generation_config.get("num_generations", 1))

    for prompt_batch in _batched(prompts, batch_size):
        texts = [_prompt_to_text(tokenizer, prompt) for prompt in prompt_batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(generation_config["max_prompt_length"]),
            return_tensors="pt",
        )
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        prompt_len = tokenized["input_ids"].shape[1]
        allowed_args = set(model.generation_config.to_dict().keys()) | {"max_new_tokens"}
        gen_kwargs = {
            "max_new_tokens": int(generation_config["max_completion_length"]),
            "do_sample": generation_config.get("do_sample"),
            "temperature": generation_config.get("temperature"),
            "top_p": generation_config.get("top_p"),
            "top_k": generation_config.get("top_k"),
            "min_p": generation_config.get("min_p"),
            "repetition_penalty": generation_config.get("repetition_penalty"),
            "num_return_sequences": num_generations,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if gen_kwargs.get("temperature") is None:
            gen_kwargs.pop("temperature", None)
        else:
            gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is None:
            gen_kwargs.pop("top_p", None)
        else:
            gen_kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("top_k") is None:
            gen_kwargs.pop("top_k", None)
        else:
            gen_kwargs["top_k"] = int(gen_kwargs["top_k"])
        if gen_kwargs.get("min_p") is None:
            gen_kwargs.pop("min_p", None)
        else:
            gen_kwargs["min_p"] = float(gen_kwargs["min_p"])
        if gen_kwargs.get("repetition_penalty") is None:
            gen_kwargs.pop("repetition_penalty", None)
        else:
            gen_kwargs["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
        if gen_kwargs.get("do_sample") is None:
            gen_kwargs.pop("do_sample", None)
        else:
            gen_kwargs["do_sample"] = bool(gen_kwargs["do_sample"])
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in allowed_args}
        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                **gen_kwargs,
            )

        for seq_idx, seq in enumerate(outputs):
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            completions.append([{"content": text}])

    return completions


def _iter_generation_configs(base_config: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    temperatures = _as_list(base_config.get("temperature"))
    top_ps = _as_list(base_config.get("top_p"))
    top_ks = _as_list(base_config.get("top_k"))
    min_ps = _as_list(base_config.get("min_p"))
    for temp, top_p, top_k, min_p in itertools.product(temperatures, top_ps, top_ks, min_ps):
        cfg = dict(base_config)
        cfg["temperature"] = temp
        cfg["top_p"] = top_p
        cfg["top_k"] = top_k
        cfg["min_p"] = min_p
        yield cfg


def _output_filename_for_config(generation_config: Dict[str, Any]) -> str:
    temp = _format_param(generation_config.get("temperature"))
    top_p = _format_param(generation_config.get("top_p"))
    top_k = _format_param(generation_config.get("top_k"))
    min_p = _format_param(generation_config.get("min_p"))
    return f"GEN-{temp}-{top_p}-{top_k}-{min_p}.jsonl"


def run_generation() -> None:
    hf_cli_login()
    dataset = load_wildguard_train(
        num_samples=NUM_SAMPLES,
        max_tokens=GENERATION_CONFIG["max_prompt_length"],
    )
    dataset = attach_prompts(dataset)

    prompts = dataset["prompt"]
    solutions = dataset["solution"]

    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    for generation_config in _iter_generation_configs(GENERATION_CONFIG):
        start_time = time.time()
        completions = generate_completions(model, tokenizer, prompts, generation_config)
        rewards = format_reward(completions)
        group_size = int(generation_config.get("num_generations", 1))
        reward_mads = []
        if group_size > 0:
            for start in range(0, len(rewards), group_size):
                group = rewards[start : start + group_size]
                if len(group) == group_size:
                    reward_mads.append(_mean_absolute_deviation(group))
        reward_mad = sum(reward_mads) / float(len(reward_mads)) if reward_mads else 0.0
        elapsed = time.time() - start_time
        print(
            f"Generated {len(completions)} samples in {elapsed:.1f}s "
            f"for temp={generation_config['temperature']} top_p={generation_config['top_p']} "
            f"top_k={generation_config['top_k']} min_p={generation_config['min_p']}"
        )

        payload = {
            "completions": completions,
            "rewards": {"format_reward": rewards},
            "prompts": list(prompts),
            "solutions": list(solutions),
            "generation_config": dict(generation_config),
            "reward_mad": reward_mad,
        }
        output_filename = _output_filename_for_config(generation_config)
        _write_reward_batch_log(OUTPUT_DIR_BASE, output_filename, payload)


def main() -> None:
    run_generation()


if __name__ == "__main__":
    main()
