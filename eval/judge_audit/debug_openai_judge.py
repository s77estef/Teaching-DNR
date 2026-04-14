#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from eval.judge_audit.common import build_messages_for_mode, load_jsonl, write_json, write_jsonl
from fine_tune.shared import extract_label_if_any, get_label_regex  # noqa: E402


DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_outputs"
DEFAULT_DATASET = Path(__file__).resolve().parent / "real_prompts_synthetic_reasoning_v1.jsonl"

_SCORE_RE = re.compile(r"<score>\s*([01](?:\.\d+)?)\s*</score>", re.IGNORECASE)
_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone debug tool for inspecting exact OpenAI judge inputs and outputs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--mode",
        choices=["rubric_only", "gold_direction", "judge_plus_accuracy"],
        default="judge_plus_accuracy",
    )
    parser.add_argument(
        "--harmful-prompts",
        type=int,
        default=1,
        help="How many distinct harmful prompts to include.",
    )
    parser.add_argument(
        "--unharmful-prompts",
        type=int,
        default=1,
        help="How many distinct unharmful prompts to include.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "perfect_articulate",
            "articulate_wrong_label",
            "adequate_but_improvable",
            "insufficient",
        ],
        help="Variant names to include for each selected prompt.",
    )
    parser.add_argument(
        "--output-prefix",
        default="openai_debug",
        help="Prefix for debug output files.",
    )
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


def _extract_normative_reasoning(text: str) -> Optional[str]:
    match = get_label_regex(
        include_normative_reasoning=True,
        capture_reasoning=True,
    ).match(text or "")
    if not match:
        return None
    reasoning = match.groupdict().get("normative_reasoning")
    return reasoning.strip() if reasoning else None


def _accuracy(text: str, gold_label: str) -> Tuple[Optional[str], float]:
    pred = extract_label_if_any(text, include_normative_reasoning=True)
    if pred is None:
        return None, 0.0
    return pred, 1.0 if pred == (gold_label or "").lower() else 0.0


def _select_debug_examples(
    rows: List[Dict[str, object]],
    *,
    harmful_prompts: int,
    unharmful_prompts: int,
    variants: List[str],
) -> List[Dict[str, object]]:
    variant_set = set(variants)
    selected: List[Dict[str, object]] = []

    for label, prompt_limit in [("harmful", harmful_prompts), ("unharmful", unharmful_prompts)]:
        prompts_seen = 0
        prompt_to_rows: Dict[str, List[Dict[str, object]]] = {}
        for row in rows:
            if row.get("gold_label") != label:
                continue
            prompt = str(row.get("prompt") or "")
            prompt_to_rows.setdefault(prompt, []).append(row)

        for prompt, prompt_rows in prompt_to_rows.items():
            by_variant = {str(row.get("variant")): row for row in prompt_rows}
            if not variant_set.issubset(by_variant.keys()):
                continue
            for variant in variants:
                selected.append(by_variant[variant])
            prompts_seen += 1
            if prompts_seen >= prompt_limit:
                break

    return selected


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    OpenAI = _import_client()
    client = OpenAI(api_key=api_key)

    rows = load_jsonl(args.dataset)
    debug_rows = _select_debug_examples(
        rows,
        harmful_prompts=args.harmful_prompts,
        unharmful_prompts=args.unharmful_prompts,
        variants=args.variants,
    )
    if not debug_rows:
        raise SystemExit("No matching debug examples found.")

    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for row in debug_rows:
        completion = str(row.get("completion") or "")
        gold_label = str(row.get("gold_label") or "").lower()
        predicted_label, accuracy = _accuracy(completion, gold_label)
        normative_reasoning = _extract_normative_reasoning(completion)
        judge_mode = "rubric_only" if args.mode == "judge_plus_accuracy" else args.mode
        messages = build_messages_for_mode(row, judge_mode)

        response = client.responses.create(
            model=args.model,
            input=messages,
        )
        raw_output = _extract_output_text(response)
        judge_score = float(_parse_score(raw_output))
        final_score = judge_score
        if args.mode == "judge_plus_accuracy":
            final_score = judge_score

        records.append(
            {
                "example_id": row.get("example_id"),
                "gold_label": gold_label,
                "variant": row.get("variant"),
                "quality_label": row.get("quality_label"),
                "prompt": row.get("prompt"),
                "completion": completion,
                "predicted_label": predicted_label,
                "accuracy": accuracy,
                "normative_reasoning": normative_reasoning,
                "mode": args.mode,
                "judge_mode_used": judge_mode,
                "judge_messages": messages,
                "judge_system_prompt": messages[0]["content"] if messages else None,
                "judge_user_prompt": messages[1]["content"] if len(messages) > 1 else None,
                "raw_output": raw_output,
                "judge_score": judge_score,
                "final_score": final_score,
                "response_id": getattr(response, "id", None),
                "judge_model": args.model,
            }
        )
        print(f"{row.get('example_id')}: raw={raw_output!r} parsed={judge_score:.3f}")

    prefix = DEBUG_OUTPUT_DIR / f"{args.output_prefix}_{args.model}_{args.mode}"
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_json(
        prefix.with_name(prefix.name + "_summary.json"),
        {
            "model": args.model,
            "mode": args.mode,
            "dataset": str(args.dataset),
            "num_examples": len(records),
            "examples": [
                {
                    "example_id": record["example_id"],
                    "variant": record["variant"],
                    "gold_label": record["gold_label"],
                    "judge_score": record["judge_score"],
                    "final_score": record["final_score"],
                    "raw_output": record["raw_output"],
                }
                for record in records
            ],
        },
    )
    print(f"Wrote detailed debug log to {prefix.with_suffix('.jsonl')}")
    print(f"Wrote summary to {prefix.with_name(prefix.name + '_summary.json')}")


if __name__ == "__main__":
    main()
