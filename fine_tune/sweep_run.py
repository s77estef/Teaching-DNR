#!/usr/bin/env python
"""
Entry point for running WandB sweeps against the supervised fine-tuning job.

wandb sweep wandb.yaml
wandb agent <sweep-id>
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import wandb

from fine_tune import fine_tune_sft_label as sft


def build_training_config_from_sweep() -> tuple[dict, sft.SFTConfig]:
    """Merge the base config with overrides coming from wandb.config."""
    sweep_cfg = wandb.config
    training_cfg = copy.deepcopy(sft.SFT_TRAINING_CONFIG)

    for key, value in sweep_cfg.items():
        if key in training_cfg:
            training_cfg[key] = value

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


def main() -> None:
    sft.hf_cli_login()
    sft.wandb_cli_login()
    wandb.init(project=os.getenv("WANDB_PROJECT"))

    train_ds, _ = sft.load_wildguard_dataset()
    train_ds = sft.attach_prompts(train_ds)
    lora_overrides = build_lora_overrides_from_sweep()
    model = sft.load_lora_model(lora_overrides)

    training_cfg, training_args = build_training_config_from_sweep()
    sft.run_trainer(model, train_ds, training_args, training_cfg)

    wandb.finish()


if __name__ == "__main__":
    main()
