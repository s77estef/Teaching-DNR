from typing import Any, List

from fine_tune.shared import extract_label_if_any
from fine_tune.train_logger import completion_to_text

_WARNED = False


def judge_reward(completions, **kwargs) -> List[float]:
    """
    Placeholder judge reward module.
    """
    global _WARNED
    if not _WARNED:
        print(
            "[reward_funcs_judge] Using placeholder judge_reward. "
            "LLM-as-a-judge scorer."
        )
        _WARNED = True

    rewards: List[float] = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        label = extract_label_if_any(text, include_normative_reasoning=True)
        rewards.append(0.1 if label is not None else 0.0)
    return rewards
