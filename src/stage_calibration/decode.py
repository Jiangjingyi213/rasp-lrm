from __future__ import annotations

from typing import Any

import torch

from src.models.hooks import model_device

from .prefill import tokenize_prompt_with_prefill
from .protocol import (
    MARKERS,
    StageTokenTracker,
    illegal_stage_tag_reason,
    marker_token_sequences,
    stage_protocol_complete,
)
from .runtime import StageMaskRuntime


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits.float() / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits = logits.masked_fill(logits < values[:, -1, None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _final_stage_start(tracker: StageTokenTracker) -> int | None:
    for start, _end, stage in reversed(tracker.marker_positions):
        if stage == "final":
            return int(start)
    return None


def _boxed_complete_after_final_marker(decoded_text: str) -> bool:
    final_index = decoded_text.rfind(MARKERS["final"])
    if final_index < 0:
        return False
    text = decoded_text[final_index + len(MARKERS["final"]) :]
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return False
    depth = 0
    opened = False
    for char in text[start + len("\\boxed") :]:
        if char == "{":
            depth += 1
            opened = True
        elif char == "}":
            depth -= 1
            if opened and depth == 0:
                return True
            if depth < 0:
                return False
    return False


@torch.no_grad()
def decode_with_stage_masks(
    model,
    tokenizer,
    prompt: str,
    runtime: StageMaskRuntime,
    *,
    prefill: str = "",
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    runtime.reset()
    tracker = StageTokenTracker(marker_token_sequences(tokenizer))
    device = model_device(model)
    inputs, _prompt_ids, prefill_ids = tokenize_prompt_with_prefill(
        tokenizer,
        prompt,
        prefill,
        max_input_tokens=max_input_tokens,
        device=device,
    )
    generated: list[int] = []
    prefill_stage: str | None = None
    for token_id in prefill_ids:
        generated.append(int(token_id))
        new_stage = tracker.feed(int(token_id))
        if new_stage:
            prefill_stage = new_stage
    outputs = model(**inputs, use_cache=True, return_dict=True)
    if prefill_stage:
        runtime.set_stage(prefill_stage)
    past = outputs.past_key_values
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    ended_with_eos = False
    stopped_after_complete_stage_answer = False
    sampled_tokens = 0
    decode_check_interval = 8
    illegal_check_window = 256
    for _ in range(max_new_tokens):
        token = _sample(outputs.logits[:, -1, :], temperature, top_p, top_k)
        token_id = int(token.item())
        sampled_tokens += 1
        generated.append(token_id)
        runtime.record_token()
        new_stage = tracker.feed(token_id)
        if new_stage:
            runtime.set_stage(new_stage)
        if tracker.fallback_reason and runtime.fallback_reason is None:
            runtime.fallback_dense(tracker.fallback_reason)
        should_check_text = sampled_tokens == 1 or sampled_tokens % decode_check_interval == 0
        if should_check_text:
            tail_ids = generated[-illegal_check_window:]
            tail_decoded = tokenizer.decode(tail_ids, skip_special_tokens=True)
            illegal_reason = illegal_stage_tag_reason(tail_decoded)
            if illegal_reason:
                tracker.fallback_dense(illegal_reason)
                runtime.fallback_dense(tracker.fallback_reason or illegal_reason)
            final_start = _final_stage_start(tracker)
            if (
                tracker.fallback_reason is None
                and final_start is not None
                and stage_protocol_complete(tracker)
            ):
                final_decoded = tokenizer.decode(generated[final_start:], skip_special_tokens=True)
                if _boxed_complete_after_final_marker(final_decoded):
                    stopped_after_complete_stage_answer = True
                    break
        if token_id in eos_ids:
            ended_with_eos = True
            break
        outputs = model(input_ids=token.unsqueeze(-1), past_key_values=past, use_cache=True, return_dict=True)
        past = outputs.past_key_values
    completion = tokenizer.decode(generated, skip_special_tokens=True).strip()
    protocol = tracker.finalize(completion)
    if protocol["fallback_reason"] and runtime.fallback_reason is None:
        runtime.fallback_dense(protocol["fallback_reason"])
    return {
        "completion": completion,
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "ended_with_eos": ended_with_eos,
        "stopped_after_complete_stage_answer": stopped_after_complete_stage_answer,
        "truncated": not ended_with_eos and sampled_tokens >= max_new_tokens,
        "stage_protocol": protocol,
        "runtime_stage_mask": runtime.summary(),
    }
