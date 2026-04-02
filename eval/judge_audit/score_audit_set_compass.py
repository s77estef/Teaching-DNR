#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List

from eval.judge_audit.common import DEFAULT_AUDIT_SET_PATH, DEFAULT_RESULTS_DIR, load_audit_rows, write_jsonl
from fine_tune.reward_funcs_judge import (
    judge_plus_accuracy_reward,
    judge_reward,
    judge_with_gold_direction_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the fixed audit set with CompassJudger.")
    parser.add_argument("--audit-set", type=Path, default=DEFAULT_AUDIT_SET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "compass_scores_v1.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["rubric_only", "gold_direction", "judge_plus_accuracy"],
        default="gold_direction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_audit_rows(args.audit_set)

    completions = [[{"content": row["completion"]}] for row in rows]
    user_prompts = [row["prompt"] for row in rows]
    gold_labels = [row["gold_label"] for row in rows]

    if args.mode == "rubric_only":
        scores = judge_reward(completions, user_prompt=user_prompts, solution=gold_labels)
    elif args.mode == "judge_plus_accuracy":
        scores = judge_plus_accuracy_reward(
            completions,
            user_prompt=user_prompts,
            solution=gold_labels,
        )
    else:
        scores = judge_with_gold_direction_reward(
            completions,
            user_prompt=user_prompts,
            solution=gold_labels,
        )

    output_rows: List[Dict[str, object]] = []
    for row, score in zip(rows, scores):
        output_rows.append(
            {
                "example_id": row["example_id"],
                "judge_name": "compass",
                "mode": args.mode,
                "judge_model": "opencompass/CompassJudger-1-1.5B-Instruct",
                "score": float(score),
            }
        )

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} Compass judge scores to {args.output}")


if __name__ == "__main__":
    main()
