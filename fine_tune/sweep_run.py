#!/usr/bin/env python
"""
Entry point for running WandB sweeps against the supervised fine-tuning job.

wandb sweep wandb.yaml
wandb agent <sweep-id>
"""

from __future__ import annotations

# at the very top of fine_tune/sweep_run.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import copy
import os
from pathlib import Path
import re

import wandb
import torch
from tqdm.auto import tqdm

from fine_tune import fine_tune_sft_label as sft


sft.TRAIN_SAMPLES = 20000
sft.TEST_SAMPLES = 1725

ANSWER_PATTERN = re.compile(r"<answer>\s*(harmful|unharmful)\s*</answer>", re.IGNORECASE)


def build_training_config_from_sweep() -> tuple[dict, sft.SFTConfig]:
    """Merge the base config with overrides coming from wandb.config."""
    sweep_cfg = wandb.config
    training_cfg = copy.deepcopy(sft.SFT_TRAINING_CONFIG)

    for key, value in sweep_cfg.items():
        if key in training_cfg:
            training_cfg[key] = value

    training_cfg["disable_tqdm"] = False

    # Unique output directory per run for easier artifact tracking
    run_id = wandb.run.name or wandb.run.id
    if run_id:
        base_dir = Path(sft.OUTPUT_DIR).parent
        training_cfg["output_dir"] = str(base_dir / f"{sft.model_name}-SFT-sweep_{run_id}")

    training_args = sft.SFTConfig(**training_cfg)
    return training_cfg, training_args


def build_lora_overrides_from_sweep() -> dict[str, float] | None:
    sweep_cfg = wandb.config
    overrides: dict[str, float] = {}
    for key in sft.LORA_CONFIG.keys():
        if key in sweep_cfg:
            overrides[key] = sweep_cfg[key]
    return overrides or None


def _extract_prediction(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).lower() if match else None


def evaluate_accuracy(model, tokenizer, seed: int = 123):
    model.eval()
    _, test_ds = sft.load_wildguard_dataset(seed=seed)
    sample_count = len(test_ds)
    prompts_ds = sft.attach_prompts(test_ds)
    device = next(model.parameters()).device

    correct = 0
    for idx in tqdm(range(sample_count), desc="Eval sweep run", unit="sample"):
        conversation = prompts_ds[idx]["prompt"]
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        prediction = _extract_prediction(generated_text)
        gold = test_ds[idx]["prompt_harm_label"].lower()
        if prediction == gold:
            correct += 1

    return correct / sample_count if sample_count else 0.0


def main() -> None:
    sft.hf_cli_login()
    sft.wandb_cli_login()
    wandb.init(project=os.getenv("WANDB_PROJECT"))

    train_ds, _ = sft.load_wildguard_dataset()
    train_ds = sft.attach_prompts(train_ds)
    lora_overrides = build_lora_overrides_from_sweep()
    model = sft.load_lora_model(lora_overrides)

    training_cfg, training_args = build_training_config_from_sweep()
    trainer = sft.run_trainer(model, train_ds, training_args, training_cfg)
    val_accuracy = evaluate_accuracy(trainer.model, trainer.tokenizer)

    # Log the metric that the sweep will optimize
    wandb.log({"eval/val_accuracy": val_accuracy})

    # Optionally, finish
    wandb.finish()


if __name__ == "__main__":
    main()
