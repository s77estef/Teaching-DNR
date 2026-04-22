#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from eval.judge_audit.common import load_jsonl, write_json, write_jsonl
from fine_tune.reward_funcs_judge import (  # noqa: E402
    ACTIVE_GOLD_DIRECTION_RUBRIC,
    ACTIVE_JUDGE_RUBRIC,
    _build_judge_messages,
    _extract_normative_reasoning,
    _get_judge_model,
    _get_judge_tokenizer,
    _parse_score,
    _resolve_device,
)
from fine_tune.shared import extract_label_if_any  # noqa: E402


DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_outputs"
DEFAULT_DATASET = Path(__file__).resolve().parent / "real_prompts_synthetic_reasoning_v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone debug tool for inspecting exact Compass judge inputs and outputs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=["rubric_only", "gold_direction", "judge_plus_accuracy"],
        default="judge_plus_accuracy",
    )
    parser.add_argument("--harmful-prompts", type=int, default=1)
    parser.add_argument("--unharmful-prompts", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "perfect_articulate",
            "articulate_wrong_label",
            "adequate_but_improvable",
            "insufficient",
        ],
    )
    parser.add_argument("--output-prefix", default="compass_debug")
    return parser.parse_args()


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
    rows = load_jsonl(args.dataset)
    debug_rows = _select_debug_examples(
        rows,
        harmful_prompts=args.harmful_prompts,
        unharmful_prompts=args.unharmful_prompts,
        variants=args.variants,
    )
    if not debug_rows:
        raise SystemExit("No matching debug examples found.")

    judge_mode = "rubric_only" if args.mode == "judge_plus_accuracy" else args.mode
    rubric = ACTIVE_JUDGE_RUBRIC if judge_mode == "rubric_only" else ACTIVE_GOLD_DIRECTION_RUBRIC

    tokenizer = _get_judge_tokenizer()
    model = _get_judge_model()
    device = _resolve_device()

    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for row in debug_rows:
        completion = str(row.get("completion") or "")
        gold_label = str(row.get("gold_label") or "").lower()
        predicted_label, accuracy = _accuracy(completion, gold_label)
        normative_reasoning = _extract_normative_reasoning(completion)
        messages = _build_judge_messages(
            user_prompt=str(row.get("prompt") or ""),
            normative_reasoning=normative_reasoning or "",
            predicted_label=predicted_label or "unharmful",
            rubric=rubric,
            gold_label=gold_label if judge_mode == "gold_direction" else None,
        )
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer([rendered], return_tensors="pt", padding=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        prompt_len = batch["attention_mask"].sum(dim=1).item()
        outputs = model.generate(
            **batch,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][prompt_len:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        judge_score = float(_parse_score(raw_output))
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
                "judge_model": "opencompass/CompassJudger-1-1.5B-Instruct",
            }
        )
        print(f"{row.get('example_id')}: raw={raw_output!r} parsed={judge_score:.3f}")

    prefix = DEBUG_OUTPUT_DIR / f"{args.output_prefix}_{args.mode}"
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_json(
        prefix.with_name(prefix.name + "_summary.json"),
        {
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
