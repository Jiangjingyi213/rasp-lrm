from __future__ import annotations

from typing import Any


def selected_action_event(event: dict[str, Any]) -> bool:
    decision = event.get("decision")
    return isinstance(decision, dict) and bool(decision.get("window_activated"))


def prior_action_candidate_events(
    events: list[dict[str, Any]],
    *,
    token_count: int,
    window_tokens: int,
    cooldown_tokens: int,
    decision_start: int,
    decision_stride: int,
    max_boundaries: int,
) -> list[dict[str, Any]]:
    if window_tokens < 1 or cooldown_tokens < 0:
        raise ValueError("On-policy window/cooldown must be valid")
    if decision_start <= 0 or decision_stride < window_tokens:
        raise ValueError("On-policy decision grid must be positive")
    if decision_start % window_tokens or decision_stride % window_tokens:
        raise ValueError("On-policy decision grid must be window-aligned")
    candidates = []
    prior_actions = 0
    last_action_boundary: int | None = None
    for event in events:
        boundary = int(event["generated_tokens"])
        if (
            boundary > 0
            and boundary >= decision_start
            and (boundary - decision_start) % decision_stride == 0
            and prior_actions > 0
            and last_action_boundary is not None
            and boundary - last_action_boundary >= window_tokens + cooldown_tokens
            and boundary + window_tokens < token_count
            and event.get("hidden_state") is not None
        ):
            candidates.append(event)
        if selected_action_event(event):
            prior_actions += 1
            last_action_boundary = boundary
    return candidates[:max_boundaries]
