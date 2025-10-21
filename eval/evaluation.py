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
    text = text.strip().lower()
    text = re.split(r"[\n\r]", text)[0]
    text = re.split(r"[.?!]", text)[0]
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
            prediction = pick_choice(raw, example.choices)
            if prediction == example.gold:
                correct += 1
            total += 1
            results.append(
                ExampleResult(
                    prompt=example.prompt,
                    gold=example.gold,
                    prediction=prediction,
                    raw_output=raw,
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
