from __future__ import annotations

import hashlib
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
    next_token_id: int | None = None
    hidden_l2_from_previous_event: float | None = None
    hidden_cosine_distance_from_previous_event: float | None = None
    hidden_state: list[float] | None = None
    logits_topk_ids: list[int] | None = None
    logits_topk_values: list[float] | None = None
    logits_l2: float | None = None
    logits_sha256: str | None = None
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


def _hidden_drift(
    previous: torch.Tensor | None, current: object | None
) -> tuple[float | None, float | None, torch.Tensor | None]:
    if current is None:
        return None, None, previous
    value = torch.as_tensor(current).detach().float().cpu().reshape(-1)
    if previous is None:
        return None, None, value
    l2 = float(torch.linalg.vector_norm(value - previous).item())
    cosine = float(
        1.0
        - torch.nn.functional.cosine_similarity(
            previous.unsqueeze(0), value.unsqueeze(0)
        ).item()
    )
    return l2, cosine, value


def _router_event(
    outputs: Any,
    observation: RuntimeObservation,
    controller: RuntimeRatioController,
    selected_ratio: float,
    previous_hidden: torch.Tensor | None,
    *,
    store_hidden_state: bool,
) -> tuple[RouterEvent, torch.Tensor | None]:
    l2, cosine, hidden = _hidden_drift(previous_hidden, observation.hidden_state)
    boundary_logits = outputs.logits[:, -1, :].detach().float()
    next_token_id = int(torch.argmax(boundary_logits, dim=-1).item())
    top_values = top_ids = None
    logits_l2 = None
    if store_hidden_state:
        top_values, top_ids = torch.topk(
            boundary_logits, k=min(8, boundary_logits.shape[-1])
        )
        logits_l2 = float(torch.linalg.vector_norm(boundary_logits).item())
        logits_sha256 = hashlib.sha256(
            boundary_logits.cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    else:
        logits_sha256 = None
    event = RouterEvent(
        generated_tokens=int(observation.generated_tokens),
        selected_ratio=float(selected_ratio),
        entropy=float(observation.entropy),
        confidence=float(observation.confidence),
        next_token_id=next_token_id,
        hidden_l2_from_previous_event=l2,
        hidden_cosine_distance_from_previous_event=cosine,
        hidden_state=hidden.tolist() if store_hidden_state and hidden is not None else None,
        logits_topk_ids=(
            [int(value) for value in top_ids[0].cpu().tolist()]
            if top_ids is not None
            else None
        ),
        logits_topk_values=(
            [float(value) for value in top_values[0].cpu().tolist()]
            if top_values is not None
            else None
        ),
        logits_l2=logits_l2,
        logits_sha256=logits_sha256,
        decision=getattr(controller, "last_decision", None),
    )
    return event, hidden


def is_affected_window_decision(step: int, window_tokens: int) -> bool:
    """Return whether a generated token decision was produced by the window action."""

    return 0 < int(step) <= int(window_tokens)


def _annotate_realized_action_windows(
    events: list[RouterEvent],
    generated_tokens: int,
    window_tokens: int,
    *,
    ended_with_eos: bool,
) -> None:
    for event in events:
        decision = event.decision
        if not isinstance(decision, dict) or not decision.get("window_activated"):
            continue
        duration = max(
            0,
            min(
                int(window_tokens),
                int(generated_tokens) - int(event.generated_tokens) - 1,
            ),
        )
        decision["realized_action_duration_tokens"] = duration
        decision["dense_restored_after_window"] = duration == int(window_tokens)
        decision["terminated_by_eos"] = bool(
            ended_with_eos and duration < int(window_tokens)
        )
        decision["action_completed_or_terminal"] = bool(
            duration == int(window_tokens) or decision["terminated_by_eos"]
        )


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
    store_router_hidden_states: bool = False,
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
    previous_event_hidden = None
    observation = _observation(outputs, generated_tokens=0)
    selected_ratio = float(controller.choose_ratio(observation))
    set_runtime_mlp_ratio(model, selected_ratio)
    event, previous_event_hidden = _router_event(
        outputs,
        observation,
        controller,
        selected_ratio,
        previous_event_hidden,
        store_hidden_state=store_router_hidden_states,
    )
    events.append(event)

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
            event, previous_event_hidden = _router_event(
                outputs,
                observation,
                controller,
                selected_ratio,
                previous_event_hidden,
                store_hidden_state=store_router_hidden_states,
            )
            events.append(event)
    _synchronize_if_cuda(device)
    decode_seconds = perf_counter() - decode_started
    _annotate_realized_action_windows(
        events,
        len(generated_ids),
        window_tokens,
        ended_with_eos=bool(generated_ids and generated_ids[-1] in eos_ids),
    )
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "completion": completion,
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "prefill_tokens": int(inputs["input_ids"].shape[1]),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "decode_tokens_per_second": len(generated_ids) / decode_seconds if decode_seconds > 0 else None,
        "router_events": [asdict(event) for event in events],
        "runtime_mlp": summarize_runtime_mlp(model),
    }


@torch.no_grad()
def greedy_decode_replay_history_counterfactual(
    model,
    tokenizer,
    prompt: str,
    forced_prefix_ids: list[int],
    action_schedule: dict[int, float],
    candidate_ratio: float,
    *,
    max_new_tokens: int = 512,
    max_input_tokens: int = 2048,
    window_tokens: int = 16,
) -> dict[str, Any]:
    """Replay a policy-induced prefix, branch one action window, then return dense."""

    if not forced_prefix_ids:
        raise ValueError("On-policy replay requires a non-empty forced prefix")
    if len(forced_prefix_ids) >= max_new_tokens:
        raise ValueError("Forced prefix must leave room for counterfactual continuation")
    if any(int(boundary) < 0 or int(boundary) % int(window_tokens) for boundary in action_schedule):
        raise ValueError("Action schedule boundaries must be non-negative and window-aligned")
    device = model_device(model)
    reset_runtime_mlp_state(model)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
    ).to(device)
    outputs = model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
    past_key_values = outputs.past_key_values
    replay_mismatches = []
    set_runtime_mlp_ratio(model, float(action_schedule.get(0, 0.0)))
    for index, token_id in enumerate(forced_prefix_ids):
        expected = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        if expected != int(token_id):
            replay_mismatches.append(
                {"generated_tokens": index, "expected": expected, "observed": int(token_id)}
            )
        token = torch.tensor([[int(token_id)]], device=device)
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        generated_tokens = index + 1
        if generated_tokens in action_schedule:
            set_runtime_mlp_ratio(model, float(action_schedule[generated_tokens]))

    boundary_observation = _observation(outputs, generated_tokens=len(forced_prefix_ids))
    boundary_logits = outputs.logits[:, -1, :].detach().float()
    boundary_top_values, boundary_top_ids = torch.topk(
        boundary_logits, k=min(8, boundary_logits.shape[-1])
    )
    boundary_next_token_id = int(boundary_top_ids[0, 0].item())
    boundary_logits_sha256 = hashlib.sha256(
        boundary_logits.cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    set_runtime_mlp_ratio(model, float(candidate_ratio))
    eos_ids = tokenizer.eos_token_id
    eos_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(item) for item in (eos_ids or [])}
    generated_ids: list[int] = []
    affected_window_ids: list[int] = []
    ended_with_eos = False
    terminated_by_eos = False
    remaining = max_new_tokens - len(forced_prefix_ids)
    for step in range(remaining):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if is_affected_window_decision(step, window_tokens):
            affected_window_ids.append(token_id)
        if token_id in eos_ids:
            ended_with_eos = True
            terminated_by_eos = len(affected_window_ids) < window_tokens
            set_runtime_mlp_ratio(model, 0.0)
            break
        outputs = model(
            input_ids=next_token.unsqueeze(-1),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if step + 1 == window_tokens:
            set_runtime_mlp_ratio(model, 0.0)
    return {
        "completion": tokenizer.decode(
            [*forced_prefix_ids, *generated_ids], skip_special_tokens=True
        ).strip(),
        "continuation": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "generated_ids": generated_ids,
        "window_ids": affected_window_ids,
        "boundary_observation": boundary_observation,
        "boundary_next_token_id": boundary_next_token_id,
        "boundary_logits_topk_ids": [
            int(value) for value in boundary_top_ids[0].cpu().tolist()
        ],
        "boundary_logits_topk_values": [
            float(value) for value in boundary_top_values[0].cpu().tolist()
        ],
        "boundary_logits_l2": float(torch.linalg.vector_norm(boundary_logits).item()),
        "boundary_logits_sha256": boundary_logits_sha256,
        "replay_mismatches": replay_mismatches,
        "dense_restored_after_window": len(affected_window_ids) == window_tokens,
        "ended_with_eos": ended_with_eos,
        "terminated_by_eos": terminated_by_eos,
        "action_completed_or_terminal": (
            len(affected_window_ids) == window_tokens or terminated_by_eos
        ),
        "runtime_mlp": summarize_runtime_mlp(model),
    }


@torch.no_grad()
def greedy_decode_single_window_counterfactual(
    model,
    tokenizer,
    prompt: str,
    forced_prefix_ids: list[int],
    ratio: float,
    *,
    max_new_tokens: int = 512,
    max_input_tokens: int = 2048,
    window_tokens: int = 16,
) -> dict[str, Any]:
    """Replay a dense prefix, prune one decode window, then return to dense."""
    if window_tokens < 1:
        raise ValueError("window_tokens must be positive")
    if len(forced_prefix_ids) >= max_new_tokens:
        raise ValueError("Forced prefix must leave room for counterfactual continuation")
    device = model_device(model)
    reset_runtime_mlp_state(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
    outputs = model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
    past_key_values = outputs.past_key_values

    set_runtime_mlp_ratio(model, 0.0)
    replay_mismatches = []
    for index, token_id in enumerate(forced_prefix_ids):
        expected = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        if expected != int(token_id):
            replay_mismatches.append(
                {"generated_tokens": index, "expected": expected, "observed": int(token_id)}
            )
        token = torch.tensor([[int(token_id)]], device=device)
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

    observation = _observation(outputs, generated_tokens=len(forced_prefix_ids))
    set_runtime_mlp_ratio(model, float(ratio))
    eos_ids = tokenizer.eos_token_id
    eos_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(item) for item in (eos_ids or [])}
    generated_ids: list[int] = []
    affected_window_ids: list[int] = []
    window_end_hidden = None
    ended_with_eos = False
    terminated_by_eos = False
    remaining = max_new_tokens - len(forced_prefix_ids)
    for step in range(remaining):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        # The first token comes from the dense boundary logits. Processing it
        # under the action affects the following token decision.
        if is_affected_window_decision(step, window_tokens):
            affected_window_ids.append(token_id)
        if token_id in eos_ids:
            ended_with_eos = True
            terminated_by_eos = len(affected_window_ids) < window_tokens
            set_runtime_mlp_ratio(model, 0.0)
            break
        outputs = model(
            input_ids=next_token.unsqueeze(-1),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if step + 1 == window_tokens:
            window_end_hidden = outputs.hidden_states[-1][:, -1, :].detach().float().cpu()
            set_runtime_mlp_ratio(model, 0.0)
    if window_end_hidden is None and getattr(outputs, "hidden_states", None):
        window_end_hidden = outputs.hidden_states[-1][:, -1, :].detach().float().cpu()
    return {
        "completion": tokenizer.decode([*forced_prefix_ids, *generated_ids], skip_special_tokens=True).strip(),
        "continuation": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "generated_ids": generated_ids,
        "window_ids": affected_window_ids,
        "window_end_hidden": window_end_hidden,
        "boundary_observation": observation,
        "replay_mismatches": replay_mismatches,
        "dense_restored_after_window": len(affected_window_ids) == window_tokens,
        "ended_with_eos": ended_with_eos,
        "terminated_by_eos": terminated_by_eos,
        "action_completed_or_terminal": (
            len(affected_window_ids) == window_tokens or terminated_by_eos
        ),
        "runtime_mlp": summarize_runtime_mlp(model),
    }
