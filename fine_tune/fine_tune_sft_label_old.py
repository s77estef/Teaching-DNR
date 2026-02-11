#!/usr/bin/env python
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy
import wandb
import sys

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fine_tune.shared import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FS,
    SYSTEM_PROMPT_NORMATIVE,
    SYSTEM_PROMPT_NNORMATIVE,
    extract_label_if_any,
    load_wildguard_train_rendered,
    load_wildguard_train,
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
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "bf16": True,
    "max_length": 256,
    "completion_only_loss": True,
    "report_to": ["wandb"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 20,
}

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


def attach_prompts(dataset: Dataset) -> Dataset:
    def make_conversation(example: Dict) -> Dict[str, List[Dict[str, str]]]:
        label = example["prompt_harm_label"].lower()
        if label not in {"harmful", "unharmful"}:
            raise ValueError(f"Unexpected prompt_harm_label: {label}")
        completion = f"<think></think><answer>{label}</answer>"
        if NORMATIVE:
            completion = f"<normative_reasoning></normative_reasoning><answer>{label}</answer>"
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ],
            "completion": [
                {"role": "assistant", "content": completion}
            ],
        }
    return dataset.map(make_conversation, remove_columns=["prompt_harm_label"])


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or SFT_TRAINING_CONFIG)
    _write_sft_config(config_for_log, training_args.output_dir)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        #formatting_func=conversation_formatter,
    )
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
    train_ds = load_wildguard_train(num_samples=TRAIN_SAMPLES)
    train_ds = attach_prompts(train_ds)
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, SFT_TRAINING_CONFIG)


if __name__ == "__main__":
    main()
