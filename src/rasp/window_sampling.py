from __future__ import annotations


def boundary_positions(
    token_count: int,
    window_tokens: int,
    max_boundaries: int | None,
    sampling: str = "prefix",
    explicit_positions: list[int] | tuple[int, ...] | None = None,
    *,
    decision_start: int | None = None,
    decision_stride: int | None = None,
    include_tail_anchor: bool = False,
) -> list[int]:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if window_tokens < 1:
        raise ValueError("window_tokens must be positive")
    if explicit_positions is not None:
        positions = [int(value) for value in explicit_positions]
        if positions != sorted(set(positions)):
            raise ValueError("Explicit boundary positions must be sorted and unique")
        if any(position < 0 or position % window_tokens != 0 for position in positions):
            raise ValueError("Explicit boundary positions must be non-negative and window-aligned")
        return [position for position in positions if position < token_count]
    if sampling == "causal_grid":
        start = int(decision_start if decision_start is not None else 2 * window_tokens)
        stride = int(decision_stride if decision_stride is not None else 2 * window_tokens)
        if start <= 0 or start % window_tokens != 0:
            raise ValueError("causal_grid decision_start must be positive and window-aligned")
        if stride < window_tokens or stride % window_tokens != 0:
            raise ValueError("causal_grid decision_stride must be at least one aligned window")
        # The action starts after the boundary logits. One dense boundary
        # token plus `window_tokens` affected decisions must remain.
        last_full_window = token_count - window_tokens - 1
        values = list(range(start, last_full_window + 1, stride))
        if include_tail_anchor and token_count - 1 >= start:
            tail = (token_count - 1) - ((token_count - 1) % window_tokens)
            if tail >= start:
                values.append(tail)
        values = sorted(set(values))
        if max_boundaries and len(values) > max_boundaries:
            raise ValueError("causal_grid does not allow truncating full-trajectory coverage")
        return values
    values = list(range(0, token_count, window_tokens))
    if not max_boundaries or len(values) <= max_boundaries:
        return values
    if sampling == "prefix":
        return values[:max_boundaries]
    if sampling != "uniform_full_trajectory":
        raise ValueError(f"Unknown aligned-window boundary sampling mode: {sampling}")
    if max_boundaries == 1:
        return [values[len(values) // 2]]
    indices = [
        round(index * (len(values) - 1) / (max_boundaries - 1))
        for index in range(max_boundaries)
    ]
    return [values[index] for index in indices]
