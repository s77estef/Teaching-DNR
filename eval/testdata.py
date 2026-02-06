"""
Extracts the 313 adversarial prompts from strongreject and lets a given model classify them.
Perfect classification would be all adversarial (label true).
NOTE: There exist two evaluators that evaluate an answer on its harmfulness. However, since we only train a classifier, this is not applicable.
"""

import importlib
import subprocess
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

try:
    from .check import check_output, hf_cli_login
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from eval.check import check_output, hf_cli_login


DATA_FILE = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
DEFAULT_PRINT_SAMPLES = 313 # is full datset


def ensure_strong_reject() -> None:
    """Install strong_reject from GitHub if it's not already available."""
    try:
        importlib.import_module("strong_reject")
    except ImportError:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/dsbowen/strong_reject@main",
        ])


def load_strongreject_prompts() -> Dataset:
    """Return the StrongReject prompts as [prompt, adversarial] pairs."""
    dataset = load_dataset("csv", data_files=DATA_FILE, split="train")
    dataset = dataset.rename_column("forbidden_prompt", "prompt")
    dataset = dataset.add_column("adversarial", [True] * len(dataset))
    drop_cols = [col for col in dataset.column_names if col not in {"prompt", "adversarial"}]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def run_check(print_samples: int | None = DEFAULT_PRINT_SAMPLES, adapter_path: str | None = None):
    dataset = load_strongreject_prompts()
    total = len(dataset)
    samples_to_print = total if print_samples is None else min(print_samples, total)
    dataset_info = {
        "name": "strongreject",
        "config": "csv",
        "split": "train",
        "sample_limit": None,
    }
    return check_output(
        print_samples=samples_to_print,
        adapter_path=adapter_path,
        test_ds=dataset,
        dataset_info=dataset_info,
    )


def main() -> None:
    ensure_strong_reject()
    hf_cli_login()
    result_path = run_check(adapter_path="fine_tune/trained_experiments/Qwen3-4B-SFT-sweep_ancient-sweep-19/checkpoint-157")
    print(f"StrongReject evaluation saved to {result_path}")


if __name__ == "__main__":
    main()
