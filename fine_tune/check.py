#!/usr/bin/env python
import time

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from fine_tune import hf_cli_login, load_wildguard_dataset, attach_prompts_sp, SYSTEM_PROMPT

# ---- config settings ----

TEST_SAMPLES = 1000
MODEL_ID = "Qwen/Qwen3-4B"
PRINT_SAMPLES=10, 
ADAPTER_PATH="fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251205_164937/checkpoint-1000"


# TODO: make dataset loading with number as argument
# TODO: make write out into json file
# - model used
# - model config for generation
# - system prompt used
# - dataset, split, number of samples -> should be full test set
# - accuracy of predicting gold
# - optional: comparison between trained and base model 

def load_wildguard_test(seed: int = 42) -> Dataset:
    test = load_dataset("allenai/wildguardmix", "wildguardtest", split="test", columns=["prompt", "adversarial"])
    test = test.shuffle(seed=seed)
    test = test.select(range(TEST_SAMPLES))
    return test

def check_output(num_samples: int = 5, adapter_path: str | None = None):
    """
    Docstring for check_output
    :param num_samples: number of samples to print into json file (accuracy still with all samples)
    :type num_samples: int
    :param adapter_path: use trained model if not none, base model if none
    :type adapter_path: str | None
    """
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base

    def generate_with_reasoning(messages):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()
        gen_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        inference_duration = end_time - start_time
        num_input_tokens = inputs['input_ids'].shape[1]
        num_generated_tokens = output_ids.shape[1] - num_input_tokens
        return generated_text, inference_duration, num_generated_tokens
    
    _, test_ds = load_wildguard_dataset()
    sample_count = min(num_samples, len(test_ds))
    raw_samples = test_ds.select(range(sample_count))
    prompts_ds = attach_prompts_sp(raw_samples)

    for idx, (prompt, info) in enumerate(zip(prompts_ds["prompt"], raw_samples), 1):
        generated_text, dt, tokens = generate_with_reasoning(prompt)
        print(f"\nSample {idx}")
        print(f"Prompt text: {info['prompt']}")
        print(f"Gold label: {info['adversarial']}")
        print(f"Model response: {generated_text.strip()}")
        print(f"Tokens: {tokens}, time: {dt:.2f}s")

def main() -> None:
    hf_cli_login()
    _, test_ds = load_wildguard_dataset()

    #check_output()
    #check_output(num_samples=10, adapter_path="fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251205_164937/checkpoint-1000")


if __name__ == "__main__":
    main()