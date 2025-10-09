"""CLI to evaluate Hugging Face LLMs on normative reasoning datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.evaluation import Evaluator
from eval.models import GenerativeModelRunner, ModelConfig
from eval.tasks import create_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["ethics", "delphi"],
        required=True,
        help="Task/dataset adapter to use.",
    )
    parser.add_argument(
        "--subset",
        default="commonsense",
        help="Subset for datasets that provide multiple tasks (e.g., ETHICS).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local path compatible with Hugging Face Transformers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 selects greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device mapping passed to Transformers (e.g., 'auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional torch dtype string (e.g., 'float16', 'bfloat16').",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template usage even if the tokenizer defines one.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt when using chat templates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump detailed JSON results.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    task_kwargs = {}
    if args.task == "ethics":
        task_kwargs["subset"] = args.subset
    task = create_task(args.task, **task_kwargs)

    model_config = ModelConfig(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        torch_dtype=args.torch_dtype,
        use_chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
    )
    runner = GenerativeModelRunner(model_config)
    evaluator = Evaluator(runner)
    result = evaluator.evaluate(task, split=args.split)

    print(
        f"Task: {result.task_name} | Split: {result.split} | Accuracy: {result.accuracy:.3f} "
        f"({result.correct}/{result.total})"
    )

    if args.output:
        payload = {
            "task": result.task_name,
            "split": result.split,
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total,
            "examples": [
                {
                    "prompt": r.prompt,
                    "gold": r.gold,
                    "prediction": r.prediction,
                    "raw_output": r.raw_output,
                    "metadata": r.metadata,
                }
                for r in result.details
            ],
            "model_config": model_config.__dict__,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
