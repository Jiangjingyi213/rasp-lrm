from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch

from src.models.hooks import model_device
from src.rasp.budget_controller import RuntimeObservation, RuntimeRatioController
from src.rasp.mlp_runtime import reset_runtime_mlp_state, set_runtime_mlp_ratio, summarize_runtime_mlp


@dataclass
class RouterEvent:
    generated_tokens: int
    selected_ratio: float
    entropy: float
    confidence: float
    decision: dict[str, Any] | None = None


def _next_token_stats(logits: torch.Tensor) -> tuple[float, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    confidence = probs.max(dim=-1).values
    return float(entropy.item()), float(confidence.item())


def _observation(outputs: Any, generated_tokens: int) -> RuntimeObservation:
    logits = outputs.logits[:, -1, :]
    entropy, confidence = _next_token_stats(logits)
    hidden_state = None
    if getattr(outputs, "hidden_states", None):
        hidden_state = outputs.hidden_states[-1][:, -1, :].detach().float().cpu()
    return RuntimeObservation(
        generated_tokens=generated_tokens,
        entropy=entropy,
        confidence=confidence,
        hidden_state=hidden_state,
    )


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def greedy_decode_runtime(
    model,
    tokenizer,
    prompt: str,
    controller: RuntimeRatioController,
    *,
    max_new_tokens: int = 512,
    max_input_tokens: int = 2048,
    window_tokens: int = 16,
) -> dict[str, Any]:
    if window_tokens < 1:
        raise ValueError("window_tokens must be positive")
    device = model_device(model)
    reset_runtime_mlp_state(model)
    if hasattr(controller, "reset"):
        controller.reset()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)

    _synchronize_if_cuda(device)
    started = perf_counter()
    outputs = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    _synchronize_if_cuda(device)
    prefill_seconds = perf_counter() - started

    generated_ids: list[int] = []
    events: list[RouterEvent] = []
    observation = _observation(outputs, generated_tokens=0)
    selected_ratio = float(controller.choose_ratio(observation))
    set_runtime_mlp_ratio(model, selected_ratio)
    events.append(
        RouterEvent(
            generated_tokens=0,
            selected_ratio=selected_ratio,
            entropy=observation.entropy,
            confidence=observation.confidence,
            decision=getattr(controller, "last_decision", None),
        )
    )

    eos_ids = tokenizer.eos_token_id
    eos_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(item) for item in (eos_ids or [])}
    past_key_values = outputs.past_key_values
    decode_started = perf_counter()
    for _step in range(max_new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if token_id in eos_ids:
            break

        outputs = model(
            input_ids=next_token.unsqueeze(-1),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if len(generated_ids) % window_tokens == 0:
            observation = _observation(outputs, generated_tokens=len(generated_ids))
            selected_ratio = float(controller.choose_ratio(observation))
            set_runtime_mlp_ratio(model, selected_ratio)
            events.append(
                RouterEvent(
                    generated_tokens=len(generated_ids),
                    selected_ratio=selected_ratio,
                    entropy=observation.entropy,
                    confidence=observation.confidence,
                    decision=getattr(controller, "last_decision", None),
                )
            )
    _synchronize_if_cuda(device)
    decode_seconds = perf_counter() - decode_started
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "completion": completion,
        "generated_tokens": len(generated_ids),
        "prefill_tokens": int(inputs["input_ids"].shape[1]),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "decode_tokens_per_second": len(generated_ids) / decode_seconds if decode_seconds > 0 else None,
        "router_events": [asdict(event) for event in events],
        "runtime_mlp": summarize_runtime_mlp(model),
    }
