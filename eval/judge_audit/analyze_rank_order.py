#!/usr/bin/env python
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from eval.judge_audit.common import load_jsonl, write_json


IDEAL_ORDERS = {
    "gold_direction": [
        "perfect_articulate",
        "adequate_but_improvable",
        "articulate_wrong_label",
        "insufficient",
    ],
    "judge_plus_accuracy": [
        "perfect_articulate",
        "articulate_wrong_label",
        "adequate_but_improvable",
        "insufficient",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize how well judge scores match the ideal rank ordering of reasoning variants."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mode", choices=["gold_direction", "judge_plus_accuracy"], required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dense_ranks(score_by_variant: Dict[str, float], ideal_order: List[str]) -> Dict[str, int]:
    groups: Dict[float, List[str]] = defaultdict(list)
    for variant, score in score_by_variant.items():
        groups[score].append(variant)

    ordered_scores = sorted(groups.keys(), reverse=True)
    ranks: Dict[str, int] = {}
    current_rank = 1
    for score in ordered_scores:
        tied_variants = sorted(
            groups[score],
            key=lambda variant: ideal_order.index(variant) if variant in ideal_order else 999,
        )
        for variant in tied_variants:
            ranks[variant] = current_rank
        current_rank += 1
    return ranks


def _pairwise_accuracy(
    score_by_variant: Dict[str, float],
    ideal_order: List[str],
) -> float:
    ideal_pos = {variant: idx for idx, variant in enumerate(ideal_order)}
    variants = [variant for variant in ideal_order if variant in score_by_variant]
    total = 0.0
    correct = 0.0
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            left = variants[i]
            right = variants[j]
            left_score = score_by_variant[left]
            right_score = score_by_variant[right]
            total += 1.0
            if left_score > right_score:
                correct += 1.0
            elif left_score == right_score:
                correct += 0.5
    return correct / total if total else 0.0


def _ranking_from_scores(score_by_variant: Dict[str, float], ideal_order: List[str]) -> List[str]:
    return sorted(
        score_by_variant.keys(),
        key=lambda variant: (
            -score_by_variant[variant],
            ideal_order.index(variant) if variant in ideal_order else 999,
        ),
    )


def main() -> None:
    args = parse_args()
    dataset_rows = {str(row["example_id"]): row for row in load_jsonl(args.dataset)}
    score_rows = [row for row in load_jsonl(args.scores) if row.get("mode") == args.mode]

    ideal_order = IDEAL_ORDERS[args.mode]
    joined: List[Dict[str, object]] = []
    for score_row in score_rows:
        example_id = str(score_row["example_id"])
        example = dataset_rows.get(example_id)
        if example is None:
            continue
        joined.append(
            {
                **example,
                "score": float(score_row["score"]),
            }
        )

    by_prompt: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in joined:
        by_prompt[str(row["prompt"])].append(row)

    per_prompt: List[Dict[str, object]] = []
    score_values_by_variant: Dict[str, List[float]] = defaultdict(list)
    rank_values_by_variant: Dict[str, List[float]] = defaultdict(list)

    for prompt, rows in by_prompt.items():
        score_by_variant = {str(row["variant"]): float(row["score"]) for row in rows}
        actual_order = _ranking_from_scores(score_by_variant, ideal_order)
        exact_match = actual_order == [v for v in ideal_order if v in score_by_variant]
        pairwise = _pairwise_accuracy(score_by_variant, ideal_order)
        ranks = _dense_ranks(score_by_variant, ideal_order)

        for variant, score in score_by_variant.items():
            score_values_by_variant[variant].append(score)
            rank_values_by_variant[variant].append(float(ranks[variant]))

        representative = rows[0]
        per_prompt.append(
            {
                "prompt": prompt,
                "gold_label": representative.get("gold_label"),
                "actual_order": actual_order,
                "ideal_order": [v for v in ideal_order if v in score_by_variant],
                "exact_match": exact_match,
                "pairwise_accuracy": pairwise,
                "scores_by_variant": {
                    variant: score_by_variant.get(variant) for variant in ideal_order if variant in score_by_variant
                },
                "ranks_by_variant": {
                    variant: ranks.get(variant) for variant in ideal_order if variant in ranks
                },
            }
        )

    summary = {
        "mode": args.mode,
        "num_prompts": len(per_prompt),
        "ideal_order": ideal_order,
        "mean_exact_match": _mean([1.0 if row["exact_match"] else 0.0 for row in per_prompt]),
        "mean_pairwise_accuracy": _mean([float(row["pairwise_accuracy"]) for row in per_prompt]),
        "mean_score_by_variant": {
            variant: _mean(values) for variant, values in sorted(score_values_by_variant.items())
        },
        "mean_rank_by_variant": {
            variant: _mean(values) for variant, values in sorted(rank_values_by_variant.items())
        },
        "per_prompt": per_prompt,
    }
    write_json(args.summary_output, summary)
    print(f"Wrote rank summary to {args.summary_output}")


if __name__ == "__main__":
    main()
