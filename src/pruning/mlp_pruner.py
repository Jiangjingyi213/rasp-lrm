from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from src.models.hooks import get_decoder_layers
from src.rasp.activation_ranker import keep_mask_from_ranking, rank_intermediate_neurons


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
        handles.append(
            mlp.register_forward_hook(
                lambda _m, _inp, out: (torch.zeros_like(out[0]), *out[1:]) if isinstance(out, tuple) else torch.zeros_like(out)
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def mlp_channel_mask(model, layer_ids: list[int], ratio: float) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        width = tensor.shape[-1]
        prune = max(1, int(round(width * ratio)))
        masked = tensor.clone()
        masked[..., :prune] = 0
        return (masked, *output[1:]) if isinstance(output, tuple) else masked

    for layer_id in layer_ids:
        mlp = _find_mlp(layers[layer_id])
        handles.append(mlp.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def mlp_intermediate_channel_mask(model, layer_ids: list[int], ratio: float) -> Iterator[None]:
    """Mask prompt-ranked FFN intermediate neurons during continuation decode.

    This is the deployment-aligned logical-mask action used to collect Runtime
    RASP-Zero training data. Prefill remains dense and establishes the ranking;
    subsequent single-token decode calls apply a nested intermediate mask.
    """

    layers = get_decoder_layers(model)
    handles = []

    def make_hook():
        ranking = None

        def hook(_module, inputs):
            nonlocal ranking
            intermediate = inputs[0]
            if intermediate.shape[1] > 1 or ranking is None:
                ranking = rank_intermediate_neurons(intermediate.detach())
                return inputs
            mask = keep_mask_from_ranking(ranking, ratio).to(
                device=intermediate.device,
                dtype=intermediate.dtype,
            )
            return (intermediate * mask, *inputs[1:])

        return hook

    for layer_id in layer_ids:
        mlp = _find_mlp(layers[layer_id])
        handles.append(mlp.down_proj.register_forward_pre_hook(make_hook()))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
