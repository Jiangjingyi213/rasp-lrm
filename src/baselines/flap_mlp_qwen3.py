from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.models.hooks import get_decoder_layers, model_device


@dataclass
class FlapMlpSummary:
    method: str
    source: str
    metric: str
    structure: str
    ratio: float
    calibration_dataset: str
    calibration_samples: int
    target: str
    physical_pruning: bool
    bias_compensation: bool
    total_layers: int
    pruned_layers: list[int]
    original_intermediate_size: int
    kept_channels_per_layer: dict[str, int]
    pruned_channels_per_layer: dict[str, int]


class BiasGPT:
    """FLAP BiasGPT statistics for one Linear layer input.

    This follows `external_repos/FLAP/lib/layerwrapper.py`: it records the
    mean input and either input fluctuation or input norm statistics for a
    projection layer. We use it on Qwen3 `mlp.down_proj`, whose input channels
    are the MLP intermediate neurons.
    """

    def __init__(self, layer: nn.Linear, metric: str) -> None:
        self.layer = layer
        self.dev = layer.weight.device
        self.in_dim = layer.weight.data.shape[1]
        self.metric = metric
        self.nsamples = 0
        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.metric == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.metric == "WIFN":
            inp = inp.float()
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = torch.zeros_like(self.baseline_inp)
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += (
                    torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1)
                    / (self.nsamples + batch_size)
                )
        self.nsamples += batch_size

    def metric_values(self) -> torch.Tensor:
        weight = self.layer.weight.data.float()
        if self.metric == "IFV":
            return self.fluc_inp.float()
        if self.metric == "WIFV":
            return self.fluc_inp.float() * torch.sum(weight.pow(2), dim=0)
        if self.metric == "WIFN":
            return (torch.abs(weight) * torch.sqrt(self.scaler_inp.float().reshape((1, -1)))).mean(dim=0)
        raise ValueError(f"Unsupported FLAP metric: {self.metric}")

    def mean_input(self) -> torch.Tensor:
        return self.baseline_inp.detach().float()


def _qwen3_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
        raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj/act_fn")
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


def _masks_from_metrics(metrics: dict[int, torch.Tensor], ratio: float, structure: str) -> dict[int, torch.Tensor]:
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"FLAP pruning ratio must be in [0, 1), got {ratio}")
    if structure not in {"UL-UM", "AL-AM"}:
        raise ValueError("FLAP-MLP Qwen3 currently supports structure in {'UL-UM', 'AL-AM'}")

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
def collect_flap_mlp_statistics_qwen3(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    metric: str,
    calibration_samples: int,
    max_input_tokens: int,
    layers: Iterable[int] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    decoder_layers = get_decoder_layers(model)
    selected_layers = _layer_ids(model, layers)
    stats = {layer_id: BiasGPT(_qwen3_mlp(decoder_layers[layer_id]).down_proj, metric) for layer_id in selected_layers}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: nn.Module, inputs, _output) -> None:
            stats[layer_id].add_batch(inputs[0].data)

        return hook

    for layer_id in selected_layers:
        handles.append(_qwen3_mlp(decoder_layers[layer_id]).down_proj.register_forward_hook(make_hook(layer_id)))

    try:
        for prompt in tqdm(calibration_texts[:calibration_samples], desc="flap-mlp-calibration"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            inputs = inputs.to(model_device(model))
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    metrics = {layer_id: stat.metric_values().detach().cpu() for layer_id, stat in stats.items()}
    means = {layer_id: stat.mean_input().detach().cpu() for layer_id, stat in stats.items()}
    return metrics, means


@torch.no_grad()
def _prune_mlp_channels(mlp: nn.Module, keep_mask: torch.Tensor, mean_input: torch.Tensor, bias_compensation: bool) -> None:
    device = mlp.gate_proj.weight.device
    keep_idx = torch.where(keep_mask)[0].to(device)
    remove_mask = (~keep_mask).to(device)
    original_down_weight = mlp.down_proj.weight.data.detach().clone()

    if bias_compensation and bool(remove_mask.any()):
        removed_mean = mean_input.to(device=device, dtype=original_down_weight.dtype) * remove_mask.to(dtype=original_down_weight.dtype)
        output_bias = removed_mean @ original_down_weight.t()
        existing_bias = mlp.down_proj.bias.data if mlp.down_proj.bias is not None else torch.zeros_like(output_bias)
        new_bias = existing_bias + output_bias
    else:
        new_bias = mlp.down_proj.bias.data.detach().clone() if mlp.down_proj.bias is not None else None

    mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.data.index_select(0, keep_idx).contiguous())
    mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.data.index_select(0, keep_idx).contiguous())
    mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.data.index_select(1, keep_idx).contiguous())
    mlp.gate_proj.out_features = int(keep_idx.numel())
    mlp.up_proj.out_features = int(keep_idx.numel())
    mlp.down_proj.in_features = int(keep_idx.numel())
    if new_bias is not None:
        mlp.down_proj.bias = nn.Parameter(new_bias.contiguous())
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(keep_idx.numel())


@torch.no_grad()
def apply_flap_mlp_pruning_qwen3(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    ratio: float,
    calibration_dataset: str = "wikitext2",
    metric: str = "WIFV",
    structure: str = "AL-AM",
    calibration_samples: int = 32,
    max_input_tokens: int = 2048,
    layers: Iterable[int] | None = None,
    bias_compensation: bool = True,
) -> FlapMlpSummary:
    metric = metric.upper()
    structure = structure.upper()
    if metric not in {"IFV", "WIFV", "WIFN"}:
        raise ValueError("FLAP-MLP Qwen3 metric must be one of {'IFV', 'WIFV', 'WIFN'}")

    decoder_layers = get_decoder_layers(model)
    selected_layers = _layer_ids(model, layers)
    first_mlp = _qwen3_mlp(decoder_layers[selected_layers[0]])
    original_intermediate_size = int(first_mlp.down_proj.weight.shape[1])

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    metrics, means = collect_flap_mlp_statistics_qwen3(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        metric=metric,
        calibration_samples=calibration_samples,
        max_input_tokens=max_input_tokens,
        layers=selected_layers,
    )
    if use_cache is not None:
        model.config.use_cache = use_cache

    masks = _masks_from_metrics(metrics, ratio=ratio, structure=structure)
    kept: dict[str, int] = {}
    pruned: dict[str, int] = {}
    for layer_id in selected_layers:
        keep_mask = masks[layer_id]
        kept[str(layer_id)] = int(keep_mask.sum().item())
        pruned[str(layer_id)] = int((~keep_mask).sum().item())
        _prune_mlp_channels(_qwen3_mlp(decoder_layers[layer_id]), keep_mask, means[layer_id], bias_compensation)

    return FlapMlpSummary(
        method="flap_mlp_qwen3",
        source="FLAP MLP port from external_repos/FLAP commit 3bb57db3449dd2fa04a5c2192de80e87e33be2b1",
        metric=metric,
        structure=structure,
        ratio=float(ratio),
        calibration_dataset=calibration_dataset,
        calibration_samples=min(int(calibration_samples), len(calibration_texts)),
        target="mlp.down_proj_input_channels",
        physical_pruning=True,
        bias_compensation=bool(bias_compensation),
        total_layers=len(decoder_layers),
        pruned_layers=selected_layers,
        original_intermediate_size=original_intermediate_size,
        kept_channels_per_layer=kept,
        pruned_channels_per_layer=pruned,
    )


def summary_to_dict(summary: FlapMlpSummary) -> dict:
    return asdict(summary)
