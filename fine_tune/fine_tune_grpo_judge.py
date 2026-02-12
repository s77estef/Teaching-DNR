#!/usr/bin/env python
import argparse
import copy
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import fine_tune.fine_tune_grpo_label as base
from fine_tune.shared import validate_and_extract_label
from fine_tune.train_logger import RewardLogger, completion_to_text


def format_gate_reward(completions, **_):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        format_ok, _ = validate_and_extract_label(
            text,
            include_normative_reasoning=base.NORMATIVE,
        )
        rewards.append(1.0 if format_ok else 0.0)
    return rewards


def build_training_args(*, reward_mode: str) -> GRPOConfig:
    training_config = copy.deepcopy(base.GRPO_TRAINING_CONFIG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = base.MODEL_ID.split("/", 1)[-1]
    training_config["output_dir"] = str(
        Path(__file__).resolve().parent
        / "trained_experiments"
        / f"{model_name}-GRPO-{reward_mode}_{timestamp}"
    )
    return GRPOConfig(**training_config)


def _load_judge_func(module_name: str, func_name: str) -> Callable:
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(
            f"Judge function not found or not callable: {module_name}.{func_name}"
        )
    return func


def _resolve_reward_funcs(
    *,
    reward_mode: str,
    judge_module: str,
    judge_func_name: str,
) -> List[Callable]:
    if reward_mode == "baseline":
        return [base.accuracy_reward]

    judge_reward = _load_judge_func(judge_module, judge_func_name)
    if reward_mode == "judge":
        return [format_gate_reward, judge_reward]
    if reward_mode == "hybrid":
        return [format_gate_reward, base.accuracy_reward, judge_reward]
    raise ValueError(f"Unsupported reward mode: {reward_mode}")


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
    config_for_log["reward_mode"] = [getattr(f, "__name__", "reward_func") for f in reward_funcs]
    base._write_grpo_config(config_for_log, training_args.output_dir)
    base._write_reward_source(training_args.output_dir)

    save_steps = training_args.save_steps or config_for_log.get("save_steps")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "judge", "hybrid"],
        default="baseline",
        help="baseline=accuracy only, judge=format gate + judge, hybrid=all rewards",
    )
    parser.add_argument(
        "--judge-module",
        default="fine_tune.reward_funcs_judge",
        help="Python module path that contains judge reward function",
    )
    parser.add_argument(
        "--judge-func",
        default="judge_reward",
        help="Function name inside --judge-module",
    )
    args = parser.parse_args()

    base.hf_cli_login()
    base.wandb_cli_login()

    train_ds = base.load_wildguard_train_rendered(
        num_samples=base.TRAIN_SAMPLES,
        max_prompt_tokens=base.GRPO_TRAINING_CONFIG["max_prompt_length"],
        tokenizer_name=base.MODEL_ID,
        system_prompt=base.SYSTEM_PROMPT,
    )

    reward_funcs = _resolve_reward_funcs(
        reward_mode=args.reward_mode,
        judge_module=args.judge_module,
        judge_func_name=args.judge_func,
    )

    model = base.load_lora_model()
    training_args = build_training_args(reward_mode=args.reward_mode)
    run_trainer(
        model=model,
        dataset=train_ds,
        training_args=training_args,
        reward_funcs=reward_funcs,
        training_config=copy.deepcopy(base.GRPO_TRAINING_CONFIG),
    )


if __name__ == "__main__":
    main()
