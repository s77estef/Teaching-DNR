from __future__ import annotations

from typing import Any, Dict, Tuple

from datasets import Dataset, load_dataset

REQUIRED_COLUMNS = {"prompt", "prompt_harm_label"}
STRONGREJECT_DATA_FILE = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)


def _dataset_info(
    *,
    name: str,
    config: str | None,
    split: str | None,
    sample_limit: int | None,
    **extra: Any,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": name,
        "config": config,
        "split": split,
        "sample_limit": sample_limit,
    }
    info.update(extra)
    return info


def _select_num_samples(ds: Dataset, num_samples: int | None) -> Dataset:
    if num_samples is None:
        return ds
    return ds.select(range(min(num_samples, len(ds))))


def load_wildguardmix_test(
    *,
    num_samples: int = 1699,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset(
        "allenai/wildguardmix",
        "wildguardtest",
        split="test",
        columns=["prompt", "prompt_harm_label"],
    )
    ds = ds.filter(lambda ex: ex["prompt_harm_label"] is not None)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name="allenai/wildguardmix",
        config="wildguardtest",
        split="test",
        sample_limit=num_samples,
    )
    return ds, info


def load_strongreject_test(
    *,
    num_samples: int = 313,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset("csv", data_files=STRONGREJECT_DATA_FILE, split="train")
    ds = ds.rename_column("forbidden_prompt", "prompt")
    ds = ds.add_column("prompt_harm_label", ["harmful"] * len(ds))
    drop_cols = [col for col in ds.column_names if col not in REQUIRED_COLUMNS]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name="strongreject",
        config="csv",
        split="train",
        sample_limit=num_samples,
        source=STRONGREJECT_DATA_FILE,
        expected_label="harmful",
    )
    return ds, info


def load_coconot_contrast_test(
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset("allenai/coconot", "contrast", split="test")
    ds = ds.add_column("prompt_harm_label", ["unharmful"] * len(ds))
    drop_cols = [col for col in ds.column_names if col not in REQUIRED_COLUMNS]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name="allenai/coconot",
        config="contrast",
        split="test",
        sample_limit=num_samples,
        expected_label="unharmful",
    )
    return ds, info


REGISTRY = {
    "wildguardmix_test": load_wildguardmix_test,
    "strongreject_test": load_strongreject_test,
    "coconot_contrast_test": load_coconot_contrast_test,
}


def get_test_dataset(
    key: str,
    *,
    num_samples: int,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    try:
        loader = REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown dataset key: {key}. Available: {available}") from exc
    ds, info = loader(num_samples=num_samples, seed=seed)
    info.setdefault("key", key)
    return ds, info
