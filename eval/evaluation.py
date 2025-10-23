"""Evaluation loop for normative reasoning tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import GenerativeModelRunner
from .tasks import BaseTask, Example


@dataclass
class ExampleResult:
    prompt: str
    gold: str
    prediction: str
    raw_output: str
    reasoning: Optional[str]
    metadata: Dict[str, str]


@dataclass
class EvalResult:
    task_name: str
    split: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    details: List[ExampleResult]


def normalize_response(text: str) -> str:
    # Trim whitespace and only look at the first line of the response.
    text = text.strip()
    text = re.split(r"[\n\r]", text)[0].strip()

    # Remove common special tokens emitted by chat models (e.g., <|im_end|>).
    text = re.sub(r"<\|.*?\|>", "", text).strip()

    # Preserve numeric answers (e.g., -1.0) so we do not drop decimal points.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return text.lower()

    text = text.lower()
    text = re.sub(r"[.?!]+$", "", text)
    return text.strip()


def pick_choice(response: str, choices: List[str]) -> str:
    normalized = normalize_response(response)

    # Prefer exact label matches before falling back to substring heuristics.
    for choice in choices:
        if normalized == choice.lower():
            return choice

    for choice in choices:
        if re.search(rf"\b{re.escape(choice.lower())}\b", normalized):
            return choice

    return normalized


def extract_reasoning_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract reasoning and answer segments from the model output."""
    reasoning = None
    answer = None

    reasoning_match = re.search(
        r"reasoning:\s*(.*?)(?:\n\s*answer:|\Z)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    answer_match = re.search(r"answer:\s*(.*)", text, flags=re.IGNORECASE)
    if answer_match:
        # Only keep the content up to the first line break or sentence end.
        answer = re.split(r"[\r\n]", answer_match.group(1))[0].strip()

    return reasoning, answer


class Evaluator:
    """Runs a model on a task and computes simple accuracy."""

    def __init__(self, model_runner: GenerativeModelRunner) -> None:
        self.model_runner = model_runner

    def evaluate(self, task: BaseTask, split: str = "validation") -> EvalResult:
        results: List[ExampleResult] = []
        correct = 0
        total = 0
        for example in task.load_split(split):
            raw = self.model_runner.generate(example.prompt)
            reasoning, answer_text = extract_reasoning_and_answer(raw)
            target_text = answer_text or raw
            prediction = pick_choice(target_text, example.choices)
            if prediction == example.gold:
                correct += 1
            total += 1
            results.append(
                ExampleResult(
                    prompt=example.prompt,
                    gold=example.gold,
                    prediction=prediction,
                    raw_output=raw,
                    reasoning=reasoning,
                    metadata=example.metadata,
                )
            )
        return EvalResult(
            task_name=task.name,
            split=split,
            total=total,
            correct=correct,
            details=results,
        )
