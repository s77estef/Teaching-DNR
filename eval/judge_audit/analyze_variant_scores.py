#!/usr/bin/env python
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.judge_audit.common import load_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize judge scores by synthetic reasoning variant and gold label."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["rubric_only", "gold_direction", "judge_plus_accuracy"],
        default="gold_direction",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    dataset_rows = {str(row["example_id"]): row for row in load_jsonl(args.dataset)}
    score_rows = [row for row in load_jsonl(args.scores) if row.get("mode") == args.mode]

    joined = []
    for score_row in score_rows:
        example_id = str(score_row["example_id"])
        example = dataset_rows.get(example_id)
        if example is None:
            continue
        joined.append(
            {
                **example,
                "judge_name": score_row.get("judge_name"),
                "judge_model": score_row.get("judge_model"),
                "mode": score_row.get("mode"),
                "score": float(score_row["score"]),
            }
        )

    by_variant: Dict[str, List[float]] = defaultdict(list)
    by_variant_and_label: Dict[str, List[float]] = defaultdict(list)
    by_prompt: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in joined:
        variant = str(row["variant"])
        gold_label = str(row["gold_label"])
        by_variant[variant].append(float(row["score"]))
        by_variant_and_label[f"{gold_label}::{variant}"].append(float(row["score"]))
        by_prompt[str(row["prompt"])].append(row)

    pairwise_prompt_checks = []
    for prompt, rows in by_prompt.items():
        rows_by_variant = {str(row["variant"]): row for row in rows}
        perfect = rows_by_variant.get("perfect_articulate")
        adequate = rows_by_variant.get("adequate_but_improvable")
        insufficient = rows_by_variant.get("insufficient")
        wrong = rows_by_variant.get("articulate_wrong_label")
        if perfect and adequate and insufficient and wrong:
            pairwise_prompt_checks.append(
                {
                    "prompt": prompt,
                    "gold_label": perfect["gold_label"],
                    "perfect_score": float(perfect["score"]),
                    "adequate_score": float(adequate["score"]),
                    "insufficient_score": float(insufficient["score"]),
                    "wrong_score": float(wrong["score"]),
                    "perfect_ge_adequate": float(perfect["score"]) >= float(adequate["score"]),
                    "adequate_ge_insufficient": float(adequate["score"]) >= float(insufficient["score"]),
                    "perfect_ge_wrong": float(perfect["score"]) >= float(wrong["score"]),
                }
            )

    summary = {
        "mode": args.mode,
        "num_examples": len(joined),
        "mean_by_variant": {key: _mean(values) for key, values in sorted(by_variant.items())},
        "mean_by_gold_and_variant": {
            key: _mean(values) for key, values in sorted(by_variant_and_label.items())
        },
        "prompt_ordering_checks": {
            "num_prompts": len(pairwise_prompt_checks),
            "perfect_ge_adequate_rate": _mean(
                [1.0 if row["perfect_ge_adequate"] else 0.0 for row in pairwise_prompt_checks]
            ),
            "adequate_ge_insufficient_rate": _mean(
                [1.0 if row["adequate_ge_insufficient"] else 0.0 for row in pairwise_prompt_checks]
            ),
            "perfect_ge_wrong_rate": _mean(
                [1.0 if row["perfect_ge_wrong"] else 0.0 for row in pairwise_prompt_checks]
            ),
        },
    }
    write_json(args.summary_output, summary)
    write_csv(
        args.table_output,
        pairwise_prompt_checks,
        fieldnames=[
            "gold_label",
            "perfect_score",
            "adequate_score",
            "insufficient_score",
            "wrong_score",
            "perfect_ge_adequate",
            "adequate_ge_insufficient",
            "perfect_ge_wrong",
            "prompt",
        ],
    )
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote per-prompt table to {args.table_output}")


if __name__ == "__main__":
    main()
