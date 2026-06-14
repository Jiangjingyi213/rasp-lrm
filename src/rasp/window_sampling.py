from __future__ import annotations


def boundary_positions(
    token_count: int,
    window_tokens: int,
    max_boundaries: int | None,
    sampling: str = "prefix",
    explicit_positions: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    if explicit_positions is not None:
        positions = [int(value) for value in explicit_positions]
        if positions != sorted(set(positions)):
            raise ValueError("Explicit boundary positions must be sorted and unique")
        if any(position < 0 or position % window_tokens != 0 for position in positions):
            raise ValueError("Explicit boundary positions must be non-negative and window-aligned")
        return [position for position in positions if position < token_count]
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
