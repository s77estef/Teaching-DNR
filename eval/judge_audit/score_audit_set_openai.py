#!/usr/bin/env python
import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from eval.judge_audit.common import (
    DEFAULT_AUDIT_SET_PATH,
    DEFAULT_RESULTS_DIR,
    build_messages_for_mode,
    extract_label_if_any,
    load_audit_rows,
    write_jsonl,
)


_SCORE_RE = re.compile(r"<score>\s*([01](?:\.\d+)?)\s*</score>", re.IGNORECASE)
_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')
JUDGE_WEIGHT = 1.0
ACCURACY_WEIGHT = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the fixed audit set with an OpenAI judge.")
    parser.add_argument("--audit-set", type=Path, default=DEFAULT_AUDIT_SET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "openai_scores_v1.jsonl",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Pinned OpenAI model for audit judging. Prefer exact model IDs for reproducibility.",
    )
    parser.add_argument(
        "--mode",
        choices=["rubric_only", "gold_direction", "judge_plus_accuracy"],
        default="gold_direction",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _import_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The `openai` package is not installed in this environment. "
            "Install the repo requirements before running this script."
        ) from exc
    return OpenAI


def _extract_output_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text).strip()

    output = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _parse_score(text: str) -> float:
    match = _SCORE_RE.search(text or "")
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    json_match = _JSON_SCORE_RE.search(text or "")
    if json_match:
        return max(0.0, min(1.0, float(json_match.group(1))))

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "score" in payload:
            return max(0.0, min(1.0, float(payload["score"])))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return 0.0


def _load_existing(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    from eval.judge_audit.common import load_jsonl

    rows = load_jsonl(path)
    return {
        f"{row['example_id']}::{row.get('mode', 'unknown')}": row
        for row in rows
    }


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    OpenAI = _import_client()
    client = OpenAI(api_key=api_key)
    audit_rows = load_audit_rows(args.audit_set)
    if args.max_examples is not None:
        audit_rows = audit_rows[: args.max_examples]

    existing = {} if args.overwrite else _load_existing(args.output)
    output_rows: List[Dict[str, object]] = list(existing.values())

    for row in audit_rows:
        example_id = str(row["example_id"])
        existing_key = f"{example_id}::{args.mode}"
        if existing_key in existing:
            continue

        judge_mode = "rubric_only" if args.mode == "judge_plus_accuracy" else args.mode
        messages = build_messages_for_mode(row, judge_mode)
        response = client.responses.create(
            model=args.model,
            input=messages,
        )
        raw_output = _extract_output_text(response)
        judge_score = float(_parse_score(raw_output))
        score = judge_score
        accuracy = None
        if args.mode == "judge_plus_accuracy":
            predicted_label = extract_label_if_any(
                row.get("completion") or "",
                include_normative_reasoning=True,
            )
            gold_label = str(row.get("gold_label") or "").lower()
            accuracy = 1.0 if predicted_label is not None and predicted_label == gold_label else 0.0
            total_weight = JUDGE_WEIGHT + ACCURACY_WEIGHT
            if total_weight <= 0:
                raise SystemExit("JUDGE_WEIGHT + ACCURACY_WEIGHT must be positive.")
            score = (
                (JUDGE_WEIGHT * judge_score) + (ACCURACY_WEIGHT * accuracy)
            ) / total_weight
        output_rows.append(
            {
                "example_id": example_id,
                "judge_name": "openai",
                "mode": args.mode,
                "judge_model": args.model,
                "score": score,
                "judge_score": judge_score,
                "accuracy": accuracy,
                "raw_output": raw_output,
                "response_id": getattr(response, "id", None),
            }
        )
        print(f"{example_id}: {score:.3f}")

    output_rows.sort(key=lambda item: str(item["example_id"]))
    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} OpenAI judge scores to {args.output}")


if __name__ == "__main__":
    main()
