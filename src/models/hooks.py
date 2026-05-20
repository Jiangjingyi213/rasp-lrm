from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def get_decoder_layers(model) -> list[torch.nn.Module]:
    for path in (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return list(obj)
    raise ValueError("Could not locate decoder layers for this model architecture")


def _identity_layer_output(inputs, output):
    hidden_states = inputs[0]
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


@contextmanager
def skip_layers(model, layer_ids: list[int]) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []
    for layer_id in layer_ids:
        layer = layers[layer_id]
        handles.append(layer.register_forward_hook(lambda _m, inp, out: _identity_layer_output(inp, out)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def token_hidden_states(model, tokenizer, text: str, layer: int = -1, max_length: int = 2048) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model_device(model))
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer][0]
    return hidden.detach().float().cpu()


@torch.no_grad()
def next_token_entropy(model, tokenizer, text: str, max_length: int = 2048) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model_device(model))
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return float(entropy.item())
