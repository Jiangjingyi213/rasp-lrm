from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from src.models.hooks import get_decoder_layers


def _find_mlp(layer: torch.nn.Module) -> torch.nn.Module:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("Could not locate MLP module on layer")


@contextmanager
def mlp_zero(model, layer_ids: list[int]) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []
    for layer_id in layer_ids:
        mlp = _find_mlp(layers[layer_id])
        handles.append(mlp.register_forward_hook(lambda _m, _inp, out: torch.zeros_like(out[0]) if isinstance(out, tuple) else torch.zeros_like(out)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
