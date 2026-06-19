from __future__ import annotations

from typing import Any

import torch


def tokenize_prompt_with_prefill(
    tokenizer,
    prompt: str,
    prefill: str,
    *,
    max_input_tokens: int,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    prefill_ids = [
        int(value)
        for value in tokenizer(prefill, add_special_tokens=False).input_ids
    ] if prefill else []
    prompt_max = max(1, int(max_input_tokens) - len(prefill_ids))
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max,
    )
    prompt_ids = [int(value) for value in prompt_inputs["input_ids"][0].tolist()]
    input_ids = torch.tensor([prompt_ids + prefill_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, prompt_ids, prefill_ids
