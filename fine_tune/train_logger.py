import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer, TrainerCallback


CHECKPOINT_BATCH_LOG_FILENAME = "checkpoint_batch_samples.jsonl"


def completion_to_text(completion) -> str:
    # TRL common shape: [{"content": "..."}]
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"] or "")
        if isinstance(first, str):
            return first
    if isinstance(completion, str):
        return completion
    return ""


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _normalize_list(value: Any, batch_size: int) -> List[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return list(value)
    return [value] * batch_size


def _extract_system_user(prompt_val: Any) -> Tuple[str | None, str | None]:
    if isinstance(prompt_val, list):
        system = None
        user = None
        for msg in prompt_val:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system" and system is None:
                system = msg.get("content")
            elif role == "user" and user is None:
                user = msg.get("content")
        return system, user
    if isinstance(prompt_val, dict):
        return None, prompt_val.get("content")
    return None, str(prompt_val)


class RewardLogger:
    def __init__(
        self,
        *,
        model_id: str,
        system_prompt: str,
        output_dir: str | Path,
        save_steps: int | None,
        num_funcs: int,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.output_dir = str(output_dir)
        self.save_steps = save_steps
        self.num_funcs = num_funcs

        self.current_step = 0
        self.is_main_process = True
        self.active = False
        self.logged_steps: set[int] = set()
        self.pending: Dict[int, Dict[str, Any]] = {}

        self._tokenizer: AutoTokenizer | None = None

    def _get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _render_prompt_string(self, prompt_val: Any) -> str:
        if isinstance(prompt_val, str):
            return prompt_val

        tokenizer = self._get_tokenizer()

        if isinstance(prompt_val, list):
            msgs = [msg for msg in prompt_val if isinstance(msg, dict)]
            if msgs:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )

        if isinstance(prompt_val, dict):
            role = prompt_val.get("role")
            content = prompt_val.get("content")
            if role and content is not None:
                msgs = [{"role": role, "content": content}]
            else:
                msgs = [
                    {
                        "role": "user",
                        "content": str(content) if content is not None else str(prompt_val),
                    }
                ]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(prompt_val)},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def update_step(self, step: int, is_main_process: bool) -> None:
        self.current_step = step
        self.is_main_process = is_main_process
        self.active = bool(self.save_steps) and step > 0 and step % int(self.save_steps) == 0

    def _write_reward_batch_log(self, step: int, payload: Dict[str, Any]) -> None:
        completions = payload.get("completions") or []
        if not isinstance(completions, list):
            completions = [completions]
        rewards_by_func = payload.get("rewards") or {}
        batch_size = len(completions)
        raw_prompts = payload.get("prompts")
        raw_user_prompts = payload.get("user_prompts")
        raw_solutions = payload.get("solutions")
        prompts_list = list(raw_prompts) if isinstance(raw_prompts, (list, tuple)) else None
        user_prompts_list = (
            list(raw_user_prompts) if isinstance(raw_user_prompts, (list, tuple)) else None
        )
        solutions_list = list(raw_solutions) if isinstance(raw_solutions, (list, tuple)) else None
        num_prompts = len(prompts_list) if prompts_list else 0
        group_size = (
            batch_size // num_prompts
            if num_prompts and batch_size % num_prompts == 0
            else None
        )
        prompts = _normalize_list(raw_prompts, batch_size)
        user_prompts = _normalize_list(raw_user_prompts, batch_size)
        solutions = _normalize_list(raw_solutions, batch_size)

        checkpoint_dir = Path(self.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_file = checkpoint_dir / CHECKPOINT_BATCH_LOG_FILENAME
        with output_file.open("w", encoding="utf-8") as fout:
            def _write_entry(entry: Dict[str, Any]) -> None:
                fout.write(json.dumps(entry, ensure_ascii=True, indent=2))
                fout.write("\n\n")

            system_prompt = self.system_prompt
            if prompts:
                extracted_system, _ = _extract_system_user(prompts[0])
                if extracted_system:
                    system_prompt = extracted_system
            header = {"step": step, "system_prompt": system_prompt}
            _write_entry(header)

            if batch_size == 0:
                return

            if group_size:
                prompt_keys = []
                for prompt_val in prompts_list or []:
                    _, user_prompt = _extract_system_user(prompt_val)
                    prompt_keys.extend([user_prompt] * group_size)
            else:
                prompt_keys = []
                for prompt_val in prompts:
                    _, user_prompt = _extract_system_user(prompt_val)
                    prompt_keys.append(user_prompt)

            groups: List[Tuple[int, int]] = []
            start_idx = 0
            for idx in range(1, batch_size):
                if prompt_keys[idx] != prompt_keys[idx - 1]:
                    groups.append((start_idx, idx))
                    start_idx = idx
            groups.append((start_idx, batch_size))

            for start_idx, end_idx in groups:
                prompt_val = prompts[start_idx]
                solution_val = solutions[start_idx]
                user_prompt = user_prompts[start_idx]
                if user_prompt is None:
                    _, user_prompt = _extract_system_user(prompt_val)
                rendered_string = self._render_prompt_string(prompt_val)
                completion_group = [
                    completion_to_text(item) for item in completions[start_idx:end_idx]
                ]
                reward_totals = []
                for local_idx in range(start_idx, end_idx):
                    total = 0.0
                    for rewards in rewards_by_func.values():
                        if isinstance(rewards, (list, tuple)) and local_idx < len(rewards):
                            try:
                                total += float(rewards[local_idx])
                            except (TypeError, ValueError):
                                total += 0.0
                    reward_totals.append(total)
                entry = {
                    "prompt": _safe_json_value(user_prompt),
                    "rendered_string": _safe_json_value(rendered_string),
                    "solution": _safe_json_value(solution_val),
                    "completions": completion_group,
                    "rewards": reward_totals,
                }
                _write_entry(entry)

    def wrap_reward(self, func):
        name = getattr(func, "__name__", "reward_func")

        def wrapped(completions, **kwargs):
            rewards = func(completions, **kwargs)
            if not (self.active and self.is_main_process):
                return rewards
            step = self.current_step
            if step in self.logged_steps:
                return rewards
            payload = self.pending.setdefault(
                step,
                {
                    "rewards": {},
                    "completions": completions,
                    "prompts": None,
                    "user_prompts": None,
                    "solutions": None,
                },
            )
            if payload["prompts"] is None:
                payload["prompts"] = kwargs.get("prompts") or kwargs.get("prompt")
            if payload["user_prompts"] is None:
                payload["user_prompts"] = (
                    kwargs.get("user_prompts") or kwargs.get("user_prompt")
                )
            if payload["solutions"] is None:
                payload["solutions"] = kwargs.get("solutions") or kwargs.get("solution")
            payload["rewards"][name] = rewards
            if len(payload["rewards"]) >= self.num_funcs:
                self._write_reward_batch_log(step, payload)
                self.logged_steps.add(step)
                self.pending.pop(step, None)
            return rewards

        return wrapped

    def get_callback(self) -> TrainerCallback:
        logger = self

        class _RewardBatchLogger(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                step = state.global_step + 1
                is_main = getattr(state, "is_world_process_zero", True)
                logger.update_step(step, is_main)
                return control

        return _RewardBatchLogger()

