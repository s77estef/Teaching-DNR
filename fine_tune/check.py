#!/usr/bin/env python
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from fine_tune.fine_tune_grpo_label import attach_prompts_sp, hf_cli_login, SYSTEM_PROMPT

# ---- config settings ----

MODEL_ID = "Qwen/Qwen3-4B"
PRINT_SAMPLES = 1725
TEST_SAMPLES = 1725
#ADAPTER_PATH = "fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251205_164937_195803_merged/checkpoint-3000"
#ADAPTER_PATH = "fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251206_094610/checkpoint-625"
ADAPTER_PATH = "fine_tune/trained_experiments/Qwen3-4B-GRPO-test_20251206_231028/checkpoint-625"

DATASET_NAME = "allenai/wildguardmix"
DATASET_CONFIG = "wildguardtest"
DATASET_SPLIT = "test"
RESULTS_DIR = Path(__file__).resolve().parent / "eval"
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
}

ANSWER_PATTERN = re.compile(r"<answer>\s*(true|false)\s*</answer>", re.IGNORECASE)
FORMAT_PATTERN = re.compile(
    r"<think>.*?</think>\s*<answer>\s*(true|false)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)


def load_wildguard_test(seed: int = 42) -> Dataset:
    test = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        columns=["prompt", "adversarial"],
    )
    test = test.shuffle(seed=seed)
    if TEST_SAMPLES is not None:
        sample_count = min(TEST_SAMPLES, len(test))
        test = test.select(range(sample_count))
    return test

def _extract_prediction(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).lower() if match else None


def _uses_correct_format(text: str) -> bool:
    return bool(FORMAT_PATTERN.search(text or ""))


def check_output(
    print_samples: int = PRINT_SAMPLES,
    adapter_path: str | None = ADAPTER_PATH,
    output_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Generate predictions, compute metrics, and persist a JSON report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                **GENERATION_CONFIG,
                eos_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()
        gen_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        inference_duration = end_time - start_time
        num_input_tokens = inputs["input_ids"].shape[1]
        num_generated_tokens = output_ids.shape[1] - num_input_tokens
        return generated_text, inference_duration, num_generated_tokens

    test_ds = load_wildguard_test()
    prompts_ds = attach_prompts_sp(test_ds)
    total_samples = len(test_ds)
    if print_samples is None:
        report_samples = 0
    else:
        report_samples = max(0, min(int(print_samples), total_samples))

    adapter_name = Path(adapter_path).name if adapter_path else "base"
    model_name = MODEL_ID.split("/", 1)[-1]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{model_name}-{adapter_name}_eval_{timestamp}.json"

    label_matches = 0
    format_matches = 0
    reported = []
    wrong_label_samples = []
    wrong_format_samples = []

    for idx in tqdm(range(total_samples), desc="Evaluating", unit="sample"):
        conversation = prompts_ds[idx]["prompt"]
        example = test_ds[idx]
        generated_text, dt, tokens = generate_with_reasoning(conversation)

        prediction = _extract_prediction(generated_text)
        gold_label = "true" if bool(example["adversarial"]) else "false"
        matches = prediction == gold_label
        format_ok = _uses_correct_format(generated_text)

        label_matches += int(matches)
        format_matches += int(format_ok)

        sample_entry = {
            "sample_id": idx + 1,
            "prompt": example["prompt"],
            "gold_label": gold_label,
            "predicted_label": prediction,
            "labels_match": matches,
            "format_correct": format_ok,
            "model_response": generated_text.strip(),
            "generated_tokens": tokens,
            "inference_seconds": dt,
        }

        if idx < report_samples:
            reported.append(sample_entry)

        if not matches:
            wrong_label_samples.append(sample_entry)

        if not format_ok:
            wrong_format_samples.append(sample_entry)

    accuracy = label_matches / total_samples if total_samples else 0.0
    format_accuracy = format_matches / total_samples if total_samples else 0.0

    payload = {
        "metadata": {
            "model_id": MODEL_ID,
            "adapter_path": adapter_path,
            "system_prompt": SYSTEM_PROMPT,
            "generation_config": {
                **GENERATION_CONFIG,
                "eos_token_id": tokenizer.eos_token_id,
            },
            "dataset": {
                "name": DATASET_NAME,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "sample_limit": TEST_SAMPLES,
                "total_samples": total_samples,
                "reported_samples": report_samples,
            },
            "metrics": {
                "label_accuracy": accuracy,
                "format_accuracy": format_accuracy,
            },
            "timestamp_utc": timestamp,
        },
        "samples": reported,
        "wrong_label_samples": wrong_label_samples,
        "wrong_format_samples": wrong_format_samples,
    }

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)

    print(
        f"Saved evaluation for {total_samples} samples "
        f"(reported {report_samples}) to {output_path}"
    )
    print(
        f"Label accuracy: {accuracy:.4f}, Format accuracy: {format_accuracy:.4f}"
    )

    return output_path

def main() -> None:
    hf_cli_login()
    check_output(print_samples=PRINT_SAMPLES, adapter_path=ADAPTER_PATH)


if __name__ == "__main__":
    main()
