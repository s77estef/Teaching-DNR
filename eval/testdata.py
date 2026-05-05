from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from datasets import Dataset, load_dataset

REQUIRED_COLUMNS = {"prompt", "prompt_harm_label"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNR_DATA_FILE = PROJECT_ROOT / "data_creation" / "dnr.json"
STRONGREJECT_DATA_FILE = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)
JAILBREAKBENCH_DATASET = "JailbreakBench/JBB-Behaviors"
WILDJAILBREAK_DATASET = "allenai/wildjailbreak"


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


def _attach_sample_ids(ds: Dataset) -> Dataset:
    return ds.add_column("sample_id", list(range(len(ds))))


def _first_non_empty(row: Dict[str, Any], columns: Iterable[str]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_harm_label(value: Any, fallback: str | None = None) -> str | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return fallback
    if text in {"harmful", "unsafe", "bad", "yes", "true", "1"}:
        return "harmful"
    if text in {"unharmful", "harmless", "benign", "safe", "no", "false", "0"}:
        return "unharmful"
    if "harmful" in text and "benign" not in text and "harmless" not in text:
        return "harmful"
    if "benign" in text or "harmless" in text or "safe" in text:
        return "unharmful"
    return fallback


def _infer_adversarial(row: Dict[str, Any], fallback: bool | None = None) -> bool:
    value = row.get("adversarial")
    if isinstance(value, bool):
        return value
    if value is not None and str(value).strip():
        # In WildJailbreak this column contains the adversarial prompt text.
        return True
    data_type = str(row.get("data_type", "")).lower()
    if "adversarial" in data_type:
        return True
    if "vanilla" in data_type:
        return False
    return bool(fallback)


def _dataset_items(raw_dataset: Any) -> Iterable[tuple[str, Dataset]]:
    if isinstance(raw_dataset, Dataset):
        yield "default", raw_dataset
    else:
        for split, ds in raw_dataset.items():
            yield str(split), ds


def load_wildguardmix_test(
    *,
    num_samples: int = 1699,
    seed: int = 42,
    only_adversarial: bool = False,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset(
        "allenai/wildguardmix",
        "wildguardtest",
        split="test",
        columns=["prompt", "prompt_harm_label", "adversarial"],
    )
    ds = _attach_sample_ids(ds)
    ds = ds.filter(lambda ex: ex["prompt_harm_label"] is not None)
    if only_adversarial:
        ds = ds.filter(lambda ex: ex["adversarial"] is True)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name="allenai/wildguardmix",
        config="wildguardtest",
        split="test",
        sample_limit=num_samples,
        only_adversarial=only_adversarial,
    )
    return ds, info


def load_strongreject_test(
    *,
    num_samples: int = 313,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset("csv", data_files=STRONGREJECT_DATA_FILE, split="train")
    ds = _attach_sample_ids(ds)
    ds = ds.rename_column("forbidden_prompt", "prompt")
    ds = ds.add_column("prompt_harm_label", ["harmful"] * len(ds))
    keep_cols = REQUIRED_COLUMNS | {"sample_id"}
    drop_cols = [col for col in ds.column_names if col not in keep_cols]
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


def load_dnr_test(
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    with DNR_DATA_FILE.open("r", encoding="utf-8") as fin:
        rows = json.load(fin)

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name="dnr",
        config="merged_rewritten_strongreject",
        split="test",
        sample_limit=num_samples,
        source=str(DNR_DATA_FILE),
        expected_label="harmful",
    )
    return ds, info


def load_coconot_contrast_test(
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = load_dataset("allenai/coconot", "contrast", split="test")
    ds = _attach_sample_ids(ds)
    ds = ds.add_column("prompt_harm_label", ["unharmful"] * len(ds))
    keep_cols = REQUIRED_COLUMNS | {"sample_id"}
    drop_cols = [col for col in ds.column_names if col not in keep_cols]
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


def load_jailbreakbench_test(
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    raw = load_dataset(JAILBREAKBENCH_DATASET, "behaviors")
    rows = []

    for split, ds in _dataset_items(raw):
        split_label = _normalize_harm_label(split)
        for idx, row in enumerate(ds):
            prompt = _first_non_empty(
                row,
                (
                    "prompt",
                    "goal",
                    "Goal",
                    "behavior",
                    "Behavior",
                    "instruction",
                    "query",
                ),
            )
            label = _normalize_harm_label(
                _first_non_empty(
                    row,
                    (
                        "prompt_harm_label",
                        "label",
                        "harm_label",
                        "category",
                        "Category",
                        "type",
                        "Type",
                    ),
                ),
                fallback=split_label,
            )
            if prompt is None or label is None:
                continue
            rows.append(
                {
                    "sample_id": f"jbb-{split}-{idx}",
                    "prompt": prompt,
                    "prompt_harm_label": label,
                    "adversarial": label == "harmful",
                }
            )

    if not rows:
        raise ValueError(
            f"Could not normalize any rows from {JAILBREAKBENCH_DATASET}. "
            "Check whether the dataset schema changed."
        )

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name=JAILBREAKBENCH_DATASET,
        config="behaviors",
        split="all",
        sample_limit=num_samples,
        label_mapping="harmful split/labels -> harmful; benign/safe split/labels -> unharmful",
        adversarial_mapping="harmful rows are counted as adversarial",
    )
    return ds, info


def load_wildjailbreak_test(
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    try:
        raw = load_dataset(
            WILDJAILBREAK_DATASET,
            "eval",
            delimiter="\t",
            keep_default_na=False,
        )
        loaded_config = "eval"
    except ValueError:
        raw = load_dataset(
            WILDJAILBREAK_DATASET,
            "train",
            delimiter="\t",
            keep_default_na=False,
        )
        loaded_config = "train"

    rows = []
    for split, ds in _dataset_items(raw):
        for idx, row in enumerate(ds):
            prompt = _first_non_empty(
                row,
                (
                    "adversarial",
                    "prompt",
                    "vanilla",
                    "query",
                    "instruction",
                    "request",
                    "goal",
                ),
            )
            label = _normalize_harm_label(
                _first_non_empty(
                    row,
                    (
                        "prompt_harm_label",
                        "label",
                        "harm_label",
                        "data_type",
                        "category",
                        "type",
                    ),
                )
            )
            if prompt is None or label is None:
                continue
            rows.append(
                {
                    "sample_id": f"wildjailbreak-{loaded_config}-{split}-{idx}",
                    "prompt": prompt,
                    "prompt_harm_label": label,
                    "adversarial": _infer_adversarial(row, fallback=label == "harmful"),
                }
            )

    if not rows:
        raise ValueError(
            f"Could not normalize any rows from {WILDJAILBREAK_DATASET}. "
            "Check whether the dataset schema changed."
        )

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=seed)
    ds = _select_num_samples(ds, num_samples)
    info = _dataset_info(
        name=WILDJAILBREAK_DATASET,
        config=loaded_config,
        split="all",
        sample_limit=num_samples,
        label_mapping="data_type/label values containing harmful -> harmful; benign/safe -> unharmful",
        adversarial_mapping="explicit adversarial flag/text or data_type containing adversarial",
    )
    return ds, info


REGISTRY = {
    "wildguardmix_test": load_wildguardmix_test,
    "dnr": load_dnr_test,
    "strongreject_test": load_strongreject_test,
    "coconot_contrast_test": load_coconot_contrast_test,
    "jailbreakbench": load_jailbreakbench_test,
    "wildjailbreak": load_wildjailbreak_test,
}


def get_test_dataset(
    key: str,
    *,
    num_samples: int,
    seed: int = 42,
    only_adversarial: bool = False,
) -> Tuple[Dataset, Dict[str, Any]]:
    try:
        loader = REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown dataset key: {key}. Available: {available}") from exc
    loader_kwargs = {"num_samples": num_samples, "seed": seed}
    if key == "wildguardmix_test":
        loader_kwargs["only_adversarial"] = only_adversarial
    ds, info = loader(**loader_kwargs)
    info.setdefault("key", key)
    return ds, info
