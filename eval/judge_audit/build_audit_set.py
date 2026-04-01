#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List

from eval.judge_audit.common import (
    DEFAULT_AUDIT_SET_PATH,
    bucket_summary,
    choose_examples_by_bucket,
    dedupe_examples,
    records_from_checkpoint_log,
    records_from_eval_report,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_EVAL_GLOBS = [
    "eval/check_outputs/Qwen3-4B-checkpoint-900_eval_20260331_161332.json",
    "eval/check_outputs/Qwen3-4B-checkpoint-900_eval_20260330_054202.json",
]
DEFAULT_CHECKPOINT_GLOBS = [
    "fine_tune/trained_experiments/Qwen3-4B-GRPO-rubric_with_gold_direction_20260328_151255/checkpoint-600/checkpoint_batch_samples.jsonl",
    "fine_tune/trained_experiments/Qwen3-4B-GRPO-rubric_with_gold_direction_20260328_151255/checkpoint-900/checkpoint_batch_samples.jsonl",
]
DEFAULT_QUOTAS: Dict[str, int] = {
    "high_reward_correct": 20,
    "correct_other": 30,
    "wrong_label_right_direction": 20,
    "articulate_wrong_direction": 20,
    "malformed": 10,
}


def _expand_globs(project_root: Path, patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        if "*" in pattern:
            paths.extend(sorted(project_root.glob(pattern)))
        else:
            path = project_root / pattern
            if path.exists():
                paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed judge audit set from existing artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_SET_PATH)
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=DEFAULT_AUDIT_SET_PATH.with_name("audit_set_v1_candidates.jsonl"),
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_AUDIT_SET_PATH.with_name("audit_set_v1_review.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_AUDIT_SET_PATH.with_name("audit_set_v1_summary.json"),
    )
    parser.add_argument("--eval-path", action="append", default=[])
    parser.add_argument("--checkpoint-path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    eval_patterns = args.eval_path or DEFAULT_EVAL_GLOBS
    checkpoint_patterns = args.checkpoint_path or DEFAULT_CHECKPOINT_GLOBS
    eval_paths = _expand_globs(project_root, eval_patterns)
    checkpoint_paths = _expand_globs(project_root, checkpoint_patterns)

    rows = []
    for path in eval_paths:
        rows.extend(records_from_eval_report(path))
    for path in checkpoint_paths:
        rows.extend(records_from_checkpoint_log(path))

    rows = dedupe_examples(rows)
    rows = [row for row in rows if row.get("adversarial") is True]
    selected = choose_examples_by_bucket(rows, seed=args.seed, quotas=DEFAULT_QUOTAS)

    write_jsonl(args.candidates_output, rows)
    write_jsonl(args.output, selected)
    review_rows = []
    for row in selected:
        review_rows.append(
            {
                "example_id": row["example_id"],
                "bucket": row["bucket"],
                "gold_label": row["gold_label"],
                "predicted_label": row["predicted_label"],
                "label_matches": row["label_matches"],
                "format_correct": row["format_correct"],
                "adversarial": row["adversarial"],
                "source_kind": row["source_kind"],
                "checkpoint_or_model": row["checkpoint_or_model"],
                "existing_reward": row["existing_reward"],
                "prompt": row["prompt"],
                "normative_reasoning": row["normative_reasoning"],
            }
        )
    write_csv(
        args.review_csv,
        review_rows,
        fieldnames=[
            "example_id",
            "bucket",
            "gold_label",
            "predicted_label",
            "label_matches",
            "format_correct",
            "adversarial",
            "source_kind",
            "checkpoint_or_model",
            "existing_reward",
            "prompt",
            "normative_reasoning",
        ],
    )
    write_json(
        args.summary_output,
        {
            "seed": args.seed,
            "num_candidates": len(rows),
            "num_selected": len(selected),
            "only_adversarial": True,
            "candidate_bucket_counts": bucket_summary(rows),
            "selected_bucket_counts": bucket_summary(selected),
            "eval_paths": [str(path) for path in eval_paths],
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "quotas": DEFAULT_QUOTAS,
        },
    )

    print(f"Wrote {len(rows)} candidate examples to {args.candidates_output}")
    print(f"Wrote {len(selected)} selected examples to {args.output}")
    print(f"Wrote review CSV to {args.review_csv}")
    print(f"Wrote summary JSON to {args.summary_output}")


if __name__ == "__main__":
    main()
