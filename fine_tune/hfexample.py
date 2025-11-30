#!/usr/bin/env python
import warnings
import re
from typing import Dict, List, Sequence, Tuple
import os

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ---- config settings ----

#DATASET_ID = "AI-MO/NuminaMath-TIR"
#TRAIN_SPLIT = "train[:5%]"
#TEST_SPLIT = "test[:5%]"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "Qwen2-0.5B-GRPO-test"

# ------------------------------------

SYSTEM_PROMPT = (
    "A conversation between User and Safeguard-Assistant. The user asks a possibly senstitive question, and the Assistant answers with true if adversarial or false if safe. "
    "The assistant first deliberates whether the question is safe to answer in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


def hf_cli_login() -> None:
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))


def load_math_dataset() -> Tuple[Dataset, Dataset]:
    train = load_dataset('allenai/wildguardmix', 'wildguardtrain')["train"]
    test = load_dataset('allenai/wildguardmix', 'wildguardtest')["test"]
    return train, test

# TODO: adapt to wildguard
def attach_prompts(dataset: Dataset) -> Dataset:
    def make_conversation(example: Dict) -> Dict[str, List[Dict[str, str]]]:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ]
        }

    mapped = dataset.map(make_conversation)
    return mapped.remove_columns([col for col in ["messages", "problem"] if col in mapped.column_names])


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
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def format_reward(completions: Sequence[Sequence[Dict[str, str]]], **_) -> List[float]:
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, content) else 0.0 for content in completion_contents]

# TODO: adapt to wildguard
def accuracy_reward(
    completions: Sequence[Sequence[Dict[str, str]]], **kwargs
) -> List[float]:
    warnings.filterwarnings(
        "ignore",
        message="equations=True in NormalizationConfig is deprecated",
        category=UserWarning,
    )
    latex_cfg = LatexExtractionConfig()
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[latex_cfg])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[latex_cfg])
        if gold_parsed:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def build_training_args() -> GRPOConfig:
    return GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        bf16=False, # not sure if GPU is bf16-capable
        fp16=True,  # flip to False if GPU is bf16-capable
        max_completion_length=64,
        num_generations=4,
        max_prompt_length=128,
        report_to=["tensorboard"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
    )

def run_trainer(model: torch.nn.Module, dataset: Dataset, training_args: GRPOConfig) -> GRPOTrainer:
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    return trainer


def main() -> None:
    hf_cli_login()
    train_ds, _ = load_math_dataset()
    train_ds = attach_prompts(train_ds)
    model = load_lora_model()
    trainer = run_trainer(model, train_ds, build_training_args())


if __name__ == "__main__":
    main()
