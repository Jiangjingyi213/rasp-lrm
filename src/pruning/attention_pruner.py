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


@contextmanager
def attention_head_mask(model, layer_ids: list[int], ratio: float) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        hidden = tensor.shape[-1]
        num_heads = getattr(_module, "num_heads", None) or getattr(_module, "num_attention_heads", None)
        if not num_heads or hidden % int(num_heads) != 0:
            masked = mask_last_dim(tensor, ratio)
        else:
            head_dim = hidden // int(num_heads)
            prune_heads = max(1, int(round(int(num_heads) * ratio)))
            masked = tensor.clone().reshape(*tensor.shape[:-1], int(num_heads), head_dim)
            masked[..., :prune_heads, :] = 0
            masked = masked.reshape_as(tensor)
        return (masked, *output[1:]) if isinstance(output, tuple) else masked

    for layer_id in layer_ids:
        attn = _find_attention(layers[layer_id])
        handles.append(attn.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def mask_last_dim(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    width = tensor.shape[-1]
    prune = max(1, int(round(width * ratio)))
    masked = tensor.clone()
    masked[..., :prune] = 0
    return masked
