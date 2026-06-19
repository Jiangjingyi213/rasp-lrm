from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.models.hooks import get_decoder_layers, model_device

from .protocol import STAGES
from .statistics import TokenMoments, wifv


def _mlp(layer):
    mlp = getattr(layer, "mlp", None)
    if not all(hasattr(mlp, name) for name in ("down_proj", "gate_proj", "up_proj")):
        raise ValueError("Expected Qwen3 MLP")
    return mlp


@torch.no_grad()
def collect_stage_statistics(
    model,
    tokenizer,
    trajectories: list[dict[str, Any]],
    *,
    c4_samples: int,
    max_input_tokens: int,
    forward_chunk_tokens: int = 1024,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    layers = get_decoder_layers(model)
    channels = {layer_id: int(_mlp(layer).down_proj.weight.shape[1]) for layer_id, layer in enumerate(layers)}
    sources = ("c4", "prompt_only", "trajectory", *STAGES)
    moments = {
        source: {layer_id: TokenMoments.zeros(size) for layer_id, size in channels.items()}
        for source in sources
    }
    active_masks: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, inputs, _output) -> None:
            values = inputs[0][0]
            for source, mask in active_masks.items():
                if bool(mask.any()):
                    moments[source][layer_id].update(values[mask.to(values.device)])

        return hook

    for layer_id, layer in enumerate(layers):
        handles.append(_mlp(layer).down_proj.register_forward_hook(make_hook(layer_id)))
    try:
        def run_input_ids(input_ids: torch.Tensor, full_masks: dict[str, torch.Tensor]) -> None:
            nonlocal active_masks
            chunk = max(1, int(forward_chunk_tokens))
            past_key_values = None
            seq = int(input_ids.shape[1])
            for start in range(0, seq, chunk):
                end = min(seq, start + chunk)
                active_masks = {
                    source: mask[start:end]
                    for source, mask in full_masks.items()
                }
                outputs = model(
                    input_ids=input_ids[:, start:end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

        for row in tqdm(trajectories, desc="stage-wifv-calibration"):
            prompt_ids = [int(value) for value in row["prompt_token_ids"]]
            generated_ids = [int(value) for value in row["generated_token_ids"]]
            token_stages = row["stage_protocol"]["token_stages"]
            input_ids = torch.tensor([prompt_ids + generated_ids], device=model_device(model))
            seq = input_ids.shape[1]
            prompt_end = min(len(prompt_ids), seq)
            generated_end = min(len(generated_ids), seq - prompt_end)
            active_masks = {
                "prompt_only": torch.arange(seq) < prompt_end,
                "trajectory": torch.tensor(
                    [False] * prompt_end
                    + [token_stages[index] is not None for index in range(generated_end)],
                    dtype=torch.bool,
                ),
            }
            for stage in STAGES:
                active_masks[stage] = torch.tensor(
                    [False] * prompt_end
                    + [token_stages[index] == stage for index in range(generated_end)],
                    dtype=torch.bool,
                )
            run_input_ids(input_ids, active_masks)

        if c4_samples > 0:
            try:
                c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load allenai/c4 for the C4 global mask baseline. "
                    "If the remote server can only use hf-mirror, rerun with "
                    "`HF_ENDPOINT=https://hf-mirror.com`. For a pure workflow smoke test, "
                    "set `masks.c4_samples: 0` in a temporary config, but do not use that "
                    "as the formal C4 baseline."
                ) from exc
            for row in tqdm(c4.take(c4_samples), total=c4_samples, desc="c4-wifv-calibration"):
                inputs = tokenizer(
                    str(row["text"]),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(model_device(model))
                active_masks = {"c4": torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool)}
                run_input_ids(inputs["input_ids"], active_masks)
    finally:
        for handle in handles:
            handle.remove()

    metrics = {
        source: {
            layer_id: wifv(moment, _mlp(layers[layer_id]).down_proj.weight)
            for layer_id, moment in layer_moments.items()
        }
        for source, layer_moments in moments.items()
    }
    means = {
        source: {layer_id: moment.mean().float() for layer_id, moment in layer_moments.items()}
        for source, layer_moments in moments.items()
    }
    counts = {
        source: {str(layer_id): moment.count for layer_id, moment in layer_moments.items()}
        for source, layer_moments in moments.items()
    }
    return metrics, means, {"token_counts": counts, "sources": list(sources)}
