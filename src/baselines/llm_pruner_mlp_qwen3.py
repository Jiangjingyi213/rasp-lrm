from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from src.models.hooks import get_decoder_layers


@dataclass
class LlmPrunerMlpSummary:
    method: str
    source: str
    importance: str
    structure: str
    ratio: float
    target: str
    physical_pruning: bool
    total_layers: int
    pruned_layers: list[int]
    original_intermediate_size: int
    kept_channels_per_layer: dict[str, int]
    pruned_channels_per_layer: dict[str, int]


def _qwen3_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj")
    return mlp


def _layer_ids(model: nn.Module, requested: Iterable[int] | None = None) -> list[int]:
    layers = get_decoder_layers(model)
    if requested is None:
        return list(range(len(layers)))
    ids = [int(i) for i in requested]
    for layer_id in ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(f"Layer id {layer_id} is out of range for {len(layers)} decoder layers")
    return ids


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _channel_importance(mlp: nn.Module, importance: str) -> torch.Tensor:
    gate = mlp.gate_proj.weight.data.float()
    up = mlp.up_proj.weight.data.float()
    down = mlp.down_proj.weight.data.float()
    if gate.shape[0] != up.shape[0] or gate.shape[0] != down.shape[1]:
        raise ValueError("Qwen3 MLP projection shapes do not share the same intermediate width")

    if importance == "l1":
        return gate.abs().sum(dim=1) + up.abs().sum(dim=1) + down.abs().sum(dim=0)
    if importance == "l2":
        return gate.pow(2).sum(dim=1).sqrt() + up.pow(2).sum(dim=1).sqrt() + down.pow(2).sum(dim=0).sqrt()
    if importance == "random":
        return torch.rand(gate.shape[0])
    raise ValueError("LLM-Pruner-style Qwen3 MLP importance must be one of {'l1', 'l2', 'random'}")


def _masks_from_importance(metrics: dict[int, torch.Tensor], ratio: float, structure: str) -> dict[int, torch.Tensor]:
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"LLM-Pruner-style pruning ratio must be in [0, 1), got {ratio}")
    if structure not in {"UL-UM", "AL-AM"}:
        raise ValueError("LLM-Pruner-style Qwen3 MLP currently supports structure in {'UL-UM', 'AL-AM'}")

    masks: dict[int, torch.Tensor] = {}
    if structure == "UL-UM":
        for layer_id, values in metrics.items():
            prune = min(values.numel() - 1, max(1, int(round(values.numel() * ratio))))
            prune_idx = torch.topk(values.float(), k=prune, largest=False).indices
            mask = torch.ones(values.numel(), dtype=torch.bool)
            mask[prune_idx.cpu()] = False
            masks[layer_id] = mask
        return masks

    layer_ids = list(metrics)
    stacked = torch.stack([metrics[layer_id].float().cpu() for layer_id in layer_ids])
    standardized = _standardize(stacked)
    prune = min(standardized.numel() - 1, max(1, int(round(standardized.numel() * ratio))))
    flat_prune_idx = torch.topk(standardized.reshape(-1), k=prune, largest=False).indices
    flat_mask = torch.ones(standardized.numel(), dtype=torch.bool)
    flat_mask[flat_prune_idx] = False
    matrix_mask = flat_mask.reshape_as(standardized)
    for row, layer_id in enumerate(layer_ids):
        masks[layer_id] = matrix_mask[row].clone()
    return masks


@torch.no_grad()
def _prune_mlp_channels(mlp: nn.Module, keep_mask: torch.Tensor) -> None:
    device = mlp.gate_proj.weight.device
    keep_idx = torch.where(keep_mask)[0].to(device)

    mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.data.index_select(0, keep_idx).contiguous())
    mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.data.index_select(0, keep_idx).contiguous())
    mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.data.index_select(1, keep_idx).contiguous())
    mlp.gate_proj.out_features = int(keep_idx.numel())
    mlp.up_proj.out_features = int(keep_idx.numel())
    mlp.down_proj.in_features = int(keep_idx.numel())
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(keep_idx.numel())


@torch.no_grad()
def apply_llm_pruner_mlp_pruning_qwen3(
    model: nn.Module,
    ratio: float,
    importance: str = "l2",
    structure: str = "UL-UM",
    layers: Iterable[int] | None = None,
) -> LlmPrunerMlpSummary:
    """Apply an LLM-Pruner-style static MLP-width baseline to Qwen3.

    This is a Qwen3 MLP-only port of the static group-pruning idea used by
    LLM-Pruner: one removable group is the matched intermediate neuron across
    gate_proj/up_proj/down_proj. It intentionally does not implement the full
    official LLM-Pruner dependency graph, Taylor pruning, or LoRA recovery.
    """

    importance = importance.lower()
    structure = structure.upper()
    decoder_layers = get_decoder_layers(model)
    selected_layers = _layer_ids(model, layers)
    first_mlp = _qwen3_mlp(decoder_layers[selected_layers[0]])
    original_intermediate_size = int(first_mlp.down_proj.weight.shape[1])

    metrics = {
        layer_id: _channel_importance(_qwen3_mlp(decoder_layers[layer_id]), importance).detach().cpu()
        for layer_id in selected_layers
    }
    masks = _masks_from_importance(metrics, ratio=ratio, structure=structure)

    kept: dict[str, int] = {}
    pruned: dict[str, int] = {}
    for layer_id in selected_layers:
        keep_mask = masks[layer_id]
        kept[str(layer_id)] = int(keep_mask.sum().item())
        pruned[str(layer_id)] = int((~keep_mask).sum().item())
        _prune_mlp_channels(_qwen3_mlp(decoder_layers[layer_id]), keep_mask)

    return LlmPrunerMlpSummary(
        method="llm_pruner_mlp_qwen3",
        source="LLM-Pruner-style static MLP group pruning for Qwen3; inspired by horseee/LLM-Pruner, Qwen3 MLP-only port",
        importance=importance,
        structure=structure,
        ratio=float(ratio),
        target="mlp.intermediate_channels",
        physical_pruning=True,
        total_layers=len(decoder_layers),
        pruned_layers=selected_layers,
        original_intermediate_size=original_intermediate_size,
        kept_channels_per_layer=kept,
        pruned_channels_per_layer=pruned,
    )


def summary_to_dict(summary: LlmPrunerMlpSummary) -> dict:
    return asdict(summary)
