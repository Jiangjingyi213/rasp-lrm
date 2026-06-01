from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

from src.pruning.attention_pruner import attention_head_mask, attention_zero
from src.pruning.layer_skipper import layer_skip
from src.pruning.mlp_pruner import mlp_channel_mask, mlp_intermediate_channel_mask, mlp_zero


@contextmanager
def pruning_context(model, module: str, layer_ids: list[int], ratio: float | None = None) -> Iterator[None]:
    module = module.lower()
    ratio = 1.0 if ratio is None else float(ratio)
    if module in {"layer", "layer_skip"}:
        with layer_skip(model, layer_ids):
            yield
    elif module in {"attention", "attention_block"}:
        with attention_zero(model, layer_ids):
            yield
    elif module in {"mlp", "mlp_block"}:
        with mlp_zero(model, layer_ids):
            yield
    elif module in {"attention_heads", "heads"}:
        with attention_head_mask(model, layer_ids, ratio):
            yield
    elif module in {"mlp_channels", "channels"}:
        with mlp_channel_mask(model, layer_ids, ratio):
            yield
    elif module in {"mlp_intermediate_channels", "intermediate_channels"}:
        with mlp_intermediate_channel_mask(model, layer_ids, ratio):
            yield
    elif module == "none":
        yield
    else:
        raise ValueError(f"Unsupported pruning module: {module}")


def select_layers(candidate_layers: list[int], ratio: float) -> list[int]:
    if not candidate_layers:
        return []
    count = max(1, int(round(len(candidate_layers) * float(ratio))))
    count = min(count, len(candidate_layers))
    if count == 1:
        return [candidate_layers[len(candidate_layers) // 2]]
    if count == len(candidate_layers):
        return list(candidate_layers)
    step = (len(candidate_layers) - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    selected = [candidate_layers[i] for i in indices]
    while len(selected) < count:
        for layer in candidate_layers:
            if layer not in selected:
                selected.append(layer)
                break
    return selected[:count]


@contextmanager
def stacked_pruning_context(model, specs: list[dict]) -> Iterator[None]:
    with ExitStack() as stack:
        for spec in specs:
            stack.enter_context(pruning_context(model, spec["module"], spec["layers"], spec.get("ratio")))
        yield
