from __future__ import annotations


def boundary_positions(
    token_count: int,
    window_tokens: int,
    max_boundaries: int | None,
    sampling: str = "prefix",
) -> list[int]:
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
