#!/usr/bin/env python
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from eval.judge_audit.common import (
    DEFAULT_AUDIT_SET_PATH,
    DEFAULT_RESULTS_DIR,
    load_audit_rows,
    load_jsonl,
    pearson,
    spearman,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agreement between Compass and OpenAI judge scores.")
    parser.add_argument("--audit-set", type=Path, default=DEFAULT_AUDIT_SET_PATH)
    parser.add_argument(
        "--compass-scores",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "compass_scores_v1.jsonl",
    )
    parser.add_argument(
        "--openai-scores",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "openai_scores_v1.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "agreement_report_v1.json",
    )
    parser.add_argument(
        "--disagreements-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "disagreements_v1.csv",
    )
    parser.add_argument("--mode", choices=["rubric_only", "gold_direction"], default="gold_direction")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def _load_scores_by_id(path: Path, mode: str) -> Dict[str, Dict[str, object]]:
    rows = load_jsonl(path)
    return {str(row["example_id"]): row for row in rows if row.get("mode") == mode}


def _pairwise_agreement(examples: List[Dict[str, object]]) -> Dict[str, object]:
    by_prompt: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for example in examples:
        by_prompt[str(example["prompt"])].append(example)

    total_pairs = 0
    same_preference = 0
    skipped_ties = 0
    for group in by_prompt.values():
        if len(group) < 2:
            continue
        for left_idx in range(len(group)):
            for right_idx in range(left_idx + 1, len(group)):
                left = group[left_idx]
                right = group[right_idx]
                compass_diff = float(left["compass_score"]) - float(right["compass_score"])
                openai_diff = float(left["openai_score"]) - float(right["openai_score"])
                if compass_diff == 0.0 or openai_diff == 0.0:
                    skipped_ties += 1
                    continue
                total_pairs += 1
                if (compass_diff > 0) == (openai_diff > 0):
                    same_preference += 1

    return {
        "evaluated_pairs": total_pairs,
        "skipped_ties": skipped_ties,
        "agreement": (same_preference / total_pairs) if total_pairs else 0.0,
    }


def main() -> None:
    args = parse_args()
    audit_rows = {str(row["example_id"]): row for row in load_audit_rows(args.audit_set)}
    compass_by_id = _load_scores_by_id(args.compass_scores, args.mode)
    openai_by_id = _load_scores_by_id(args.openai_scores, args.mode)

    joined: List[Dict[str, object]] = []
    for example_id, example in audit_rows.items():
        compass = compass_by_id.get(example_id)
        openai = openai_by_id.get(example_id)
        if compass is None or openai is None:
            continue
        joined.append(
            {
                **example,
                "compass_score": float(compass["score"]),
                "openai_score": float(openai["score"]),
                "abs_diff": abs(float(compass["score"]) - float(openai["score"])),
            }
        )

    compass_scores = [float(row["compass_score"]) for row in joined]
    openai_scores = [float(row["openai_score"]) for row in joined]
    exact_bucket_agreement = sum(
        1 for row in joined if float(row["compass_score"]) == float(row["openai_score"])
    )
    by_bucket: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "mean_abs_diff": 0.0})
    for row in joined:
        bucket = str(row["bucket"])
        by_bucket[bucket]["count"] += 1
        by_bucket[bucket]["mean_abs_diff"] += float(row["abs_diff"])
    for stats in by_bucket.values():
        count = int(stats["count"])
        stats["mean_abs_diff"] = stats["mean_abs_diff"] / count if count else 0.0

    joined.sort(key=lambda row: (-float(row["abs_diff"]), str(row["example_id"])))
    top_disagreements = joined[: args.top_k]
    write_csv(
        args.disagreements_csv,
        top_disagreements,
        fieldnames=[
            "example_id",
            "bucket",
            "gold_label",
            "predicted_label",
            "label_matches",
            "adversarial",
            "checkpoint_or_model",
            "existing_reward",
            "compass_score",
            "openai_score",
            "abs_diff",
            "prompt",
            "normative_reasoning",
        ],
    )
    summary = {
        "mode": args.mode,
        "num_compared_examples": len(joined),
        "pearson": pearson(compass_scores, openai_scores),
        "spearman": spearman(compass_scores, openai_scores),
        "exact_score_agreement": (exact_bucket_agreement / len(joined)) if joined else 0.0,
        "mean_abs_diff": (sum(row["abs_diff"] for row in joined) / len(joined)) if joined else 0.0,
        "pairwise_preference": _pairwise_agreement(joined),
        "bucket_summary": dict(sorted(by_bucket.items())),
        "top_disagreement_ids": [row["example_id"] for row in top_disagreements],
    }
    write_json(args.summary_output, summary)
    print(f"Wrote summary JSON to {args.summary_output}")
    print(f"Wrote top {len(top_disagreements)} disagreements to {args.disagreements_csv}")


if __name__ == "__main__":
    main()

