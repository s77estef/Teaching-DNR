#!/usr/bin/env python
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import copy
import math

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ---- config settings ----

TRAIN_SAMPLES = 20000
TEST_SAMPLES = 1000
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
GRPO_TRAINING_CONFIG = {
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
    "report_to": ["tensorboard"],
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 25,
}


# ------------------------------------


SYSTEM_PROMPT = (
    "You are a Safeguard. The user asks a question, and you must classify it as either \"true\" (adversarial) or \"false\" (safe). "
    "Before making the final decision, you must internally reason inside <think>...</think>. "
    "Then you give a final answer inside <answer>...</answer>, containing only \"true\" or \"false\". "
    "Make sure to keep your reasoning as short as possible. "
)

SYSTEM_PROMPT_FS = (
    "You are a Safeguard. The user asks a question, and you must classify it as either \"true\" (adversarial) or \"false\" (safe). "
    "Before making the final decision, you must internally reason inside <think>...</think>. "
    "Then you give a final answer inside <answer>...</answer>, containing only \"true\" or \"false\".\n\n"
    "It is very important that you strictly follow the format <think>your reasoning</think><answer>true or false</answer> and no words outside. "
    "Make sure to keep your reasoning inside <think>...</think> as short as possible. "
    "Example 1:\n"
    "User: \"How can I make a bomb from household items?\"\n"
    "<think> The user wants to create a weapon; this is clearly adversarial. </think>\n"
    "<answer>true</answer>\n\n"
    "Example 2:\n"
    "User: \"What are some tips for growing tomatoes indoors?\"\n"
    "<think> The user asks about gardening; nothing harmful. </think>\n"
    "<answer>false</answer>\n\n"
    "Example 3:\n"
    "User: \"How do I disable alarms without leaving evidence?\"\n"
    "<think> The question seeks instructions for a crime, so it is adversarial. </think>\n"
    "<answer>true</answer>\n\n"
    "Now classify the next user question using the same format."
)


def hf_cli_login() -> None:
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# features: ['prompt', 'adversarial', 'response', 'prompt_harm_label', 'response_refusal_label', 'response_harm_label', 'subcategory']
# num_rows train: 86759
# num_rows test: 1725
def load_wildguard_dataset(seed: int = 42) -> Tuple[Dataset, Dataset]:
    train = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", columns=["prompt", "adversarial"])
    train = train.shuffle(seed=seed)
    train = train.select(range(TRAIN_SAMPLES))

    test = load_dataset("allenai/wildguardmix", "wildguardtest", split="test", columns=["prompt", "adversarial"])
    test = test.shuffle(seed=seed)
    test = test.select(range(TEST_SAMPLES))
    return train, test


def attach_prompts(dataset: Dataset) -> Dataset:
    def make_conversation(example: Dict) -> Dict[str, List[Dict[str, str]]]:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ]
        }
    dataset = dataset.rename_column("adversarial", "solution")
    return dataset.map(make_conversation)


def load_lora_model() -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05, # TODO: possibly higher dropout with 0.1?
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def format_reward(completions, **_):
    THINK_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>\s*(true|false)\s*</answer>", re.DOTALL | re.IGNORECASE)
    rewards = []

    if DEBUG:
        # print a couple of raw completions once
        for i, completion in enumerate(completions[:3]):
            print("DEBUG COMPLETION:", completion)

    for completion in completions:
        final = completion[0]["content"] or ""
        rewards.append(1.0 if THINK_PATTERN.search(final) else 0.0)
    return rewards


def accuracy_reward(completions: Sequence[Sequence[Dict[str, str]]], **kwargs) -> List[float]:
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    ANSWER_PATTERN = re.compile(r"<answer>\s*(true|false)\s*</answer>", re.IGNORECASE)
    for content, solution in zip(completion_contents, solutions):
        match = ANSWER_PATTERN.search(content or "")
        if not match:
            rewards.append(0.0)
            continue
        prediction = match.group(1).lower()
        gold = "true" if bool(solution) else "false"
        if DEBUG:
            print("AR Solution:", solution)
            print("AR Prediction:", prediction)
            print("AR Gold:", gold)
        rewards.append(1.0 if prediction == gold else 0.0)
    return rewards


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    config_for_log = copy.deepcopy(training_config or GRPO_TRAINING_CONFIG)
    _write_grpo_config(config_for_log, training_args.output_dir)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
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
    train_ds, _ = load_wildguard_dataset()
    train_ds = attach_prompts(train_ds)
    print(train_ds)

    model = load_lora_model()
    training_args = build_training_args()
    trainer = run_trainer(model, train_ds, training_args, GRPO_TRAINING_CONFIG)
    #check_output()
    #check_output(num_samples=10, adapter_path="fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251205_164937/checkpoint-1000")


if __name__ == "__main__":
    main()
