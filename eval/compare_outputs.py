#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


DEFAULT_OUTPUT_ROOT = Path("eval/comparison_outputs")


@dataclass
class RunSummary:
    name: str
    path: Path
    samples: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class ReportBuffer:
    def __init__(self) -> None:
        self._parts: List[str] = []

    def line(self, text: str = "") -> None:
        self._parts.append(text)

    def render(self) -> str:
        return "\n".join(self._parts) + "\n"


def _short_name(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("Qwen3-4B-", "")
    stem = stem.replace("_eval_", "@")
    return stem


def _load_run(path: Path) -> RunSummary:
    payload = json.loads(path.read_text())
    return RunSummary(
        name=_short_name(path),
        path=path,
        samples=payload["samples"],
        metrics=payload["metadata"]["metrics"],
    )


def _sample_index(run: RunSummary) -> Dict[Any, Dict[str, Any]]:
    return {sample["sample_id"]: sample for sample in run.samples}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_ratio(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def _label_flip_counts(samples: Iterable[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        if sample.get("labels_match"):
            continue
        gold = sample.get("gold_label")
        pred = sample.get("predicted_label")
        counter[f"{gold}->{pred}"] += 1
    return counter


def _filter_run(run: RunSummary, predicate: Callable[[Dict[str, Any]], bool]) -> RunSummary:
    return RunSummary(
        name=run.name,
        path=run.path,
        samples=[sample for sample in run.samples if predicate(sample)],
        metrics=run.metrics,
    )


def _collect_sample_rows(runs: Sequence[RunSummary]) -> List[Dict[str, Any]]:
    indexes = {run.name: _sample_index(run) for run in runs}
    sample_ids = [sample["sample_id"] for sample in runs[0].samples]
    rows = []
    for sample_id in sample_ids:
        aligned = [indexes[run.name][sample_id] for run in runs]
        wrong_count = sum(not sample["labels_match"] for sample in aligned)
        format_count = sum(not sample["format_correct"] for sample in aligned)
        gold_values = {sample["gold_label"] for sample in aligned}
        prompt_values = {sample["prompt"] for sample in aligned}
        adversarial_values = {bool(sample.get("adversarial") is True) for sample in aligned}
        if len(gold_values) != 1 or len(prompt_values) != 1 or len(adversarial_values) != 1:
            raise ValueError(f"Sample {sample_id} is inconsistent across runs.")

        rows.append(
            {
                "sample_id": sample_id,
                "wrong_count": wrong_count,
                "format_count": format_count,
                "gold_label": aligned[0]["gold_label"],
                "adversarial": aligned[0].get("adversarial"),
                "prompt": aligned[0]["prompt"],
                "per_run": {
                    run.name: {
                        "predicted_label": sample.get("predicted_label"),
                        "labels_match": sample["labels_match"],
                        "format_correct": sample["format_correct"],
                    }
                    for run, sample in zip(runs, aligned)
                },
            }
        )
    return rows


def add_run_summary_section(report: ReportBuffer, runs: Sequence[RunSummary]) -> None:
    report.line("=== Run Summary ===")
    header = (
        f"{'run':<33} {'label_acc':>10} {'format_acc':>10} "
        f"{'wrong_lbl':>10} {'wrong_fmt':>10} {'top_flips':>24}"
    )
    report.line(header)
    report.line("-" * len(header))
    for run in runs:
        total = len(run.samples)
        wrong_label = sum(not sample["labels_match"] for sample in run.samples)
        wrong_format = sum(not sample["format_correct"] for sample in run.samples)
        label_acc = (total - wrong_label) / total if total else 0.0
        format_acc = (total - wrong_format) / total if total else 0.0
        top_flips = ", ".join(
            f"{flip}:{count}" for flip, count in _label_flip_counts(run.samples).most_common(2)
        )
        report.line(
            f"{run.name:<33} "
            f"{_format_ratio(label_acc):>10} "
            f"{_format_ratio(format_acc):>10} "
            f"{wrong_label:>10} {wrong_format:>10} "
            f"{top_flips[:24]:>24}"
        )
    report.line()


def add_overlap_section(report: ReportBuffer, runs: Sequence[RunSummary]) -> None:
    report.line("=== Wrong-Label Overlap (Jaccard / shared) ===")
    wrong_ids = {
        run.name: {sample["sample_id"] for sample in run.samples if not sample["labels_match"]}
        for run in runs
    }

    for left_idx, left in enumerate(runs):
        for right in runs[left_idx + 1 :]:
            left_ids = wrong_ids[left.name]
            right_ids = wrong_ids[right.name]
            shared = len(left_ids & right_ids)
            union = len(left_ids | right_ids)
            jaccard = shared / union if union else 0.0
            report.line(
                f"{left.name:<33} vs {right.name:<33} "
                f"jaccard={jaccard:0.3f} shared={shared:>3} "
                f"left_only={len(left_ids - right_ids):>3} right_only={len(right_ids - left_ids):>3}"
            )
    report.line()


def add_disagreement_summary(report: ReportBuffer, rows: Sequence[Dict[str, Any]], total_runs: int) -> None:
    counts = Counter(row["wrong_count"] for row in rows if row["wrong_count"] > 0)
    report.line("=== Error Frequency Across Runs ===")
    for wrong_count in range(total_runs, 0, -1):
        report.line(f"wrong in {wrong_count}/{total_runs} runs: {counts.get(wrong_count, 0)} samples")
    report.line()


def add_hardest_samples_section(
    report: ReportBuffer,
    rows: Sequence[Dict[str, Any]],
    run_names: Sequence[str],
    top_n: int,
    prompt_chars: int,
) -> None:
    report.line(f"=== Samples Wrong Most Often (top {top_n}) ===")
    ordered = sorted(
        (row for row in rows if row["wrong_count"] > 0),
        key=lambda row: (-row["wrong_count"], -row["format_count"], row["sample_id"]),
    )
    for row in ordered[:top_n]:
        report.line(
            f"sample_id={row['sample_id']} wrong_in={row['wrong_count']}/{len(run_names)} "
            f"format_bad_in={row['format_count']} gold={row['gold_label']} "
            f"adversarial={row['adversarial']}"
        )
        report.line(f"prompt: {_truncate(row['prompt'], prompt_chars)}")
        preds = " | ".join(
            f"{name}={row['per_run'][name]['predicted_label']}"
            f"{'' if row['per_run'][name]['labels_match'] else '*'}"
            f"{'' if row['per_run'][name]['format_correct'] else '!fmt'}"
            for name in run_names
        )
        report.line(f"preds:  {preds}")
        report.line()


def add_unique_errors_section(
    report: ReportBuffer,
    rows: Sequence[Dict[str, Any]],
    run_names: Sequence[str],
    top_n: int,
    prompt_chars: int,
) -> None:
    report.line(f"=== Unique Errors Per Run (up to {top_n} each) ===")
    for run_name in run_names:
        unique = [
            row
            for row in rows
            if not row["per_run"][run_name]["labels_match"] and row["wrong_count"] == 1
        ]
        report.line(f"{run_name}: {len(unique)} unique wrong-label samples")
        for row in unique[:top_n]:
            pred = row["per_run"][run_name]["predicted_label"]
            fmt = row["per_run"][run_name]["format_correct"]
            report.line(
                f"  sample_id={row['sample_id']} gold={row['gold_label']} "
                f"pred={pred} format_correct={fmt} adversarial={row['adversarial']}"
            )
            report.line(f"  prompt: {_truncate(row['prompt'], prompt_chars)}")
        report.line()


def build_text_report(
    title: str,
    runs: Sequence[RunSummary],
    rows: Sequence[Dict[str, Any]],
    run_names: Sequence[str],
    top_n: int,
    prompt_chars: int,
) -> str:
    report = ReportBuffer()
    report.line(title)
    report.line(f"Generated at UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    report.line(f"Runs compared: {len(runs)}")
    report.line(f"Samples in scope: {len(rows)}")
    report.line("Input files:")
    for run in runs:
        report.line(f"- {run.path}")
    report.line()

    add_run_summary_section(report, runs)
    add_overlap_section(report, runs)
    add_disagreement_summary(report, rows, total_runs=len(runs))
    add_hardest_samples_section(report, rows, run_names, top_n=top_n, prompt_chars=prompt_chars)
    add_unique_errors_section(report, rows, run_names, top_n=top_n, prompt_chars=prompt_chars)
    return report.render()


def _json_ready_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "wrong_count": row["wrong_count"],
            "format_count": row["format_count"],
            "gold_label": row["gold_label"],
            "adversarial": row["adversarial"],
            "prompt": row["prompt"],
            "per_run": row["per_run"],
        }
        for row in rows
    ]


def write_disagreement_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    run_names: Sequence[str],
) -> None:
    fieldnames = [
        "sample_id",
        "wrong_count",
        "format_count",
        "gold_label",
        "adversarial",
        "prompt",
    ]
    for run_name in run_names:
        fieldnames.extend(
            [
                f"{run_name}__predicted_label",
                f"{run_name}__labels_match",
                f"{run_name}__format_correct",
            ]
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-item["wrong_count"], item["sample_id"])):
            record = {
                "sample_id": row["sample_id"],
                "wrong_count": row["wrong_count"],
                "format_count": row["format_count"],
                "gold_label": row["gold_label"],
                "adversarial": row["adversarial"],
                "prompt": row["prompt"],
            }
            for run_name in run_names:
                per_run = row["per_run"][run_name]
                record[f"{run_name}__predicted_label"] = per_run["predicted_label"]
                record[f"{run_name}__labels_match"] = per_run["labels_match"]
                record[f"{run_name}__format_correct"] = per_run["format_correct"]
            writer.writerow(record)


def _default_output_dir() -> Path:
    timestamp = datetime.utcnow().strftime("compare_%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluation JSON files and highlight overlapping misclassifications."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["eval/to_compare/*.json"],
        help="Evaluation JSON files or glob patterns.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of examples to show in hardest-sample and unique-error sections.",
    )
    parser.add_argument(
        "--prompt-chars",
        type=int,
        default=180,
        help="Maximum prompt characters to print per sample in the text reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where report files will be written. Defaults to a timestamped folder in eval/comparison_outputs/.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the full standard summary to stdout; only print output paths.",
    )
    return parser.parse_args()


def _expand_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matched = sorted(Path().glob(value))
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(value))
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def main() -> None:
    args = parse_args()
    paths = _expand_paths(args.paths)
    if len(paths) < 2:
        raise SystemExit("Need at least two evaluation JSON files to compare.")

    runs = [_load_run(path) for path in paths]
    expected_ids = [sample["sample_id"] for sample in runs[0].samples]
    for run in runs[1:]:
        sample_ids = [sample["sample_id"] for sample in run.samples]
        if sample_ids != expected_ids:
            raise SystemExit(
                f"Sample ordering mismatch between {runs[0].path.name} and {run.path.name}."
            )

    run_names = [run.name for run in runs]
    rows = _collect_sample_rows(runs)

    adversarial_runs = [
        _filter_run(run, lambda sample: bool(sample.get("adversarial") is True)) for run in runs
    ]
    adversarial_rows = [row for row in rows if bool(row.get("adversarial") is True)]

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_report = build_text_report(
        "Compare Outputs Report",
        runs,
        rows,
        run_names,
        top_n=args.top_n,
        prompt_chars=args.prompt_chars,
    )
    adversarial_text_report = build_text_report(
        "Compare Outputs Report (Adversarial Only)",
        adversarial_runs,
        adversarial_rows,
        run_names,
        top_n=args.top_n,
        prompt_chars=args.prompt_chars,
    )

    text_path = output_dir / "summary.txt"
    adversarial_text_path = output_dir / "summary_adversarial.txt"
    json_path = output_dir / "disagreements.json"
    csv_path = output_dir / "disagreements.csv"
    meta_path = output_dir / "report_metadata.json"

    text_path.write_text(text_report)
    adversarial_text_path.write_text(adversarial_text_report)
    json_path.write_text(json.dumps(_json_ready_rows(rows), indent=2))
    write_disagreement_csv(csv_path, rows, run_names)
    meta_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "input_files": [str(path) for path in paths],
                "run_names": run_names,
                "output_files": {
                    "summary_txt": str(text_path),
                    "summary_adversarial_txt": str(adversarial_text_path),
                    "disagreements_json": str(json_path),
                    "disagreements_csv": str(csv_path),
                },
            },
            indent=2,
        )
    )

    if not args.quiet:
        print(text_report, end="")
    print(f"Wrote report files to: {output_dir}")
    print(f"- {text_path}")
    print(f"- {adversarial_text_path}")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
