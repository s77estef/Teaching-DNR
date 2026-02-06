# TODO use load train from shared

#!/usr/bin/env python
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINE_TUNE_DIR = PROJECT_ROOT / "fine_tune"
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "check_outputs"

# support running both as module and standalone script
try:
    from ..fine_tune.shared import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_NORMATIVE,
        attach_prompts_for_eval,
        hf_cli_login,
        validate_and_extract_label,
        validate_dataset_columns,
    )
    from .testdata import REQUIRED_COLUMNS, get_test_dataset
except ImportError:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from fine_tune.shared import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_NORMATIVE,
        attach_prompts_for_eval,
        hf_cli_login,
        validate_and_extract_label,
        validate_dataset_columns,
    )
    from eval.testdata import REQUIRED_COLUMNS, get_test_dataset

# ---- config settings ----

MODEL_ID = "Qwen/Qwen3-4B"
# max non-Null: 1699
PRINT_SAMPLES = 1699
TEST_SAMPLES = 1699

ADAPTER_PATH = None
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20251205_164937_195803_merged/checkpoint-3000"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20251206_094610/checkpoint-625"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20251206_231028/checkpoint-625"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-SFT-test_20251210_201043/checkpoint-157"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-SFT-sweep_ancient-sweep-19/checkpoint-157"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-SFT-sweep_youthful-terrain-41/checkpoint-157"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-SFT-sweep_ethereal-sweep-5/checkpoint-157"
ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20251218_180547-6/checkpoint-625"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260106_180254/checkpoint-225"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260107_132438/checkpoint-175"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260108_055324/checkpoint-100"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260122_155529/checkpoint-600"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260123_164321-123-512/checkpoint-1025"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260127_213505-133/checkpoint-1250"
#ADAPTER_PATH = FINE_TUNE_DIR / "trained_experiments/Qwen3-4B-GRPO-test_20260204_191019-138/checkpoint-450"



DATASET_KEY = "wildguardmix_test"
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.6,
}
NORMATIVE = True
if NORMATIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT_NORMATIVE


REASONING_TAG = "think"


def check_output(
    print_samples: int = PRINT_SAMPLES,
    adapter_path: str | None = ADAPTER_PATH,
    output_dir: Path | str = RESULTS_DIR,
    test_ds: Optional[Dataset] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    dataset_key: str = DATASET_KEY,
) -> Path:
    """Generate predictions, compute metrics, and persist a JSON report.

    If `test_ds` is provided it must expose `prompt` and `prompt_harm_label`
    columns and `dataset_info` can describe the source metadata.
    """

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
        gen_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        inference_duration = end_time - start_time
        num_input_tokens = inputs["input_ids"].shape[1]
        num_generated_tokens = output_ids.shape[1] - num_input_tokens
        return generated_text, inference_duration, num_generated_tokens

    if test_ds is None:
        test_ds, dataset_info = get_test_dataset(
            dataset_key,
            num_samples=TEST_SAMPLES,
        )
    else:
        dataset_info = dict(
            dataset_info
            or {
                "name": "custom",
                "config": None,
                "split": None,
                "sample_limit": None,
            }
        )

    validate_dataset_columns(test_ds, REQUIRED_COLUMNS)

    prompts_ds = attach_prompts_for_eval(test_ds, system_prompt=SYSTEM_PROMPT)
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

        format_ok, pred_label = validate_and_extract_label(
            generated_text, reasoning_tag=REASONING_TAG
        )
        gold_label = (example["prompt_harm_label"] or "").lower()

        # Only count label correctness if format is ok (mirrors reward gating)
        matches = bool(format_ok and pred_label is not None and pred_label == gold_label)

        label_matches += int(matches)
        format_matches += int(format_ok)

        sample_entry = {
            "sample_id": idx + 1,
            "prompt": example["prompt"],
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "labels_match": matches,
            "format_correct": format_ok,
            "model_response": (generated_text or "").strip(),
            "generated_tokens": tokens,
            "inference_seconds": dt,
        }

        if idx < report_samples:
            reported.append(sample_entry)

        if not matches:
            wrong_label_samples.append(sample_entry)

        if not format_ok:
            wrong_format_samples.append(sample_entry)

        # delete later
        if (idx != 0) and (idx % 100 == 0):
            print(f"Accuracy at {idx}: {label_matches / idx}")

    accuracy = label_matches / total_samples if total_samples else 0.0
    format_accuracy = format_matches / total_samples if total_samples else 0.0

    adapter_meta = str(adapter_path) if adapter_path is not None else None

    payload = {
        "metadata": {
            "model_id": MODEL_ID,
            "adapter_path": adapter_meta,
            "system_prompt": SYSTEM_PROMPT,
            "generation_config": {
                **GENERATION_CONFIG,
                "eos_token_id": tokenizer.eos_token_id,
            },
            "dataset": {
                **dataset_info,
                "total_samples": total_samples,
                "reported_samples": report_samples,
            },
            "metrics": {
                # "Label accuracy" is gated by format correctness, like training reward
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

    print(f"Saved evaluation for {total_samples} samples (reported {report_samples}) to {output_path}")
    print(f"Label accuracy: {accuracy:.4f}, Format accuracy: {format_accuracy:.4f}")

    return output_path


def main() -> None:
    hf_cli_login()
    check_output(
        print_samples=PRINT_SAMPLES,
        adapter_path=ADAPTER_PATH,
        dataset_key=DATASET_KEY,
    )


if __name__ == "__main__":
    main()
