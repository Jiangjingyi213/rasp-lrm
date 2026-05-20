from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from src.models.hooks import get_decoder_layers


def _find_attention(layer: torch.nn.Module) -> torch.nn.Module:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("Could not locate attention module on layer")


@contextmanager
def attention_zero(model, layer_ids: list[int]) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []
    for layer_id in layer_ids:
        attn = _find_attention(layers[layer_id])
        handles.append(attn.register_forward_hook(lambda _m, _inp, out: (torch.zeros_like(out[0]), *out[1:]) if isinstance(out, tuple) else torch.zeros_like(out)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
