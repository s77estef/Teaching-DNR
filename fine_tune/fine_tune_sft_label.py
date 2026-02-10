#!/usr/bin/env python
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy
import wandb

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from fine_tune.shared import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FS,
    SYSTEM_PROMPT_NORMATIVE,
    SYSTEM_PROMPT_NNORMATIVE,
    load_wildguard_train_rendered,
    hf_cli_login,
    wandb_cli_login,
)
from fine_tune.train_logger import SFTLogger, wrap_data_collator


# ---- config settings ----

TRAIN_SAMPLES = 10000
#MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
#MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
#MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen3-4B"
DEBUG = False
NORMATIVE = True

if NORMATIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_NNORMATIVE

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = MODEL_ID.split("/", 1)[-1]
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "trained_experiments",
    f"{model_name}-SFT-test_{timestamp}",
)
SFT_CONFIG_FILENAME = "sft_config.json"
SFT_TRAINING_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 1e-5,
    "remove_unused_columns": False,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "bf16": True,
    "max_length": 256,
    "completion_only_loss": True,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 20,
}
SFT_GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.6,
}
SFT_LOG_GENERATION_SAMPLES = -1

LORA_CONFIG: Dict[str, Any] = {
    "task_type": "CAUSAL_LM",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,  # possibly higher dropout with 0.1?
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


RAW_LOG_FIELDS = ("user_prompt", "rendered_string", "solution", "completion")


def _build_completion(label: str) -> str:
    label = (label or "").lower()
    if label not in {"harmful", "unharmful"}:
        raise ValueError(f"Unexpected prompt_harm_label: {label}")
    completion = f"<think></think><answer>{label}</answer>"
    if NORMATIVE:
        completion = f"<normative_reasoning></normative_reasoning><answer>{label}</answer>"
    return completion


def prepare_sft_dataset() -> Dataset:
    ds = load_wildguard_train_rendered(
        num_samples=TRAIN_SAMPLES,
        max_prompt_tokens=SFT_TRAINING_CONFIG["max_length"],
        tokenizer_name=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
    )
    ds = ds.rename_column("prompt", "rendered_string")

    def add_completion(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": ex["rendered_string"],
            "completion": _build_completion(ex["solution"]),
        }

    return ds.map(add_completion)


class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, sft_logger: SFTLogger | None = None, **kwargs) -> None:
        self.sft_logger = sft_logger
        self._last_raw_batch: Dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.sft_logger is not None:
            raw = {field: inputs.get(field) for field in RAW_LOG_FIELDS}
            self._last_raw_batch = raw
            for field in RAW_LOG_FIELDS:
                if field in inputs:
                    inputs.pop(field)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


def load_lora_model(lora_overrides: Dict[str, Any] | None = None) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    lora_params = copy.deepcopy(LORA_CONFIG)
    if lora_overrides:
        lora_params.update(lora_overrides)
    lora_cfg = LoraConfig(**lora_params)
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_training_args() -> SFTConfig:
    return SFTConfig(**SFT_TRAINING_CONFIG)

# only for training documentation
def _write_sft_config(config: Dict[str, Any], output_dir: str, results: Dict[str, Any] | None = None) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"config": config}
    if results:
        payload["results"] = results
    config_path = output_path / SFT_CONFIG_FILENAME
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
    training_args: SFTConfig,
    training_config: Dict[str, Any] | None = None,
) -> SFTTrainer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or SFT_TRAINING_CONFIG)
    _write_sft_config(config_for_log, training_args.output_dir)

    save_steps = training_args.save_steps or SFT_TRAINING_CONFIG.get("save_steps")
    sft_logger = SFTLogger(
        system_prompt=SYSTEM_PROMPT,
        output_dir=training_args.output_dir,
        save_steps=save_steps,
        generate_samples=SFT_LOG_GENERATION_SAMPLES,
        generation_config=SFT_GENERATION_CONFIG,
    )

    trainer = LoggingSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        sft_logger=sft_logger,
    )
    trainer.data_collator = wrap_data_collator(trainer.data_collator, RAW_LOG_FIELDS)
    trainer.add_callback(sft_logger.get_callback(trainer))
    start_time = time.time()
    trainer.train()
    #trainer.train(resume_from_checkpoint="fine_tune/trained_experiments/name_here/checkpoint-1000")
    elapsed = time.time() - start_time
    trainer.save_model(training_args.output_dir)
    final_results = {
        "final_steps": trainer.state.global_step,
        "training_time": _format_duration(elapsed),
    }
    _write_sft_config(config_for_log, training_args.output_dir, final_results)
    return trainer


def main() -> None:
    hf_cli_login()
    wandb_cli_login()
    train_ds = prepare_sft_dataset()
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, SFT_TRAINING_CONFIG)


if __name__ == "__main__":
    main()
