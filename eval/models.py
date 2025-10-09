"""Model loading and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ModelConfig:
    model_name_or_path: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 0.95
    device: str = "auto"
    torch_dtype: Optional[str] = None
    use_chat_template: bool = True
    system_prompt: Optional[str] = None


class GenerativeModelRunner:
    """Wrapper around Hugging Face autoregressive models."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        torch_dtype = getattr(torch, config.torch_dtype) if config.torch_dtype else None
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name_or_path
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=config.device,
            torch_dtype=torch_dtype,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _prepare_prompt(self, prompt: str) -> str:
        if not self.config.use_chat_template or not hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str) -> str:
        prepared_prompt = self._prepare_prompt(prompt)
        inputs = self.tokenizer(prepared_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :])
        return generated.strip()

    def batch_generate(self, prompts: Iterable[str]) -> Iterable[str]:
        for prompt in prompts:
            yield self.generate(prompt)
