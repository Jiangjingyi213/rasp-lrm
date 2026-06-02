from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import torch
from torch import nn

from src.models.hooks import get_decoder_layers
from src.rasp.activation_ranker import keep_mask_from_ranking, rank_intermediate_neurons


def _normalized_ratios(ratios: Iterable[float]) -> tuple[float, ...]:
    values = sorted({0.0, *(float(ratio) for ratio in ratios)})
    for ratio in values:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"Runtime RASP-Zero pruning ratios must be in [0, 1), got {ratio}")
    return tuple(values)


class RuntimeMaskedQwen3MLP(nn.Module):
    """Qwen3 FFN wrapper for correctness-first runtime RASP experiments.

    Prefill is always dense and refreshes a prompt-conditioned neuron ranking.
    Decode applies an intermediate-neuron mask chosen by the online controller.
    The wrapper does not claim speedup yet: gate/up/down projections remain
    dense in v0. A later reduced-weight backend can preserve this interface.
    """

    def __init__(self, original_mlp: nn.Module, ratios: Iterable[float]) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.supported_ratios = _normalized_ratios(ratios)
        self.active_ratio = 0.0
        self.ranking: torch.Tensor | None = None
        self._mask_cache: dict[float, torch.Tensor] = {}
        self.prefill_tokens = 0
        self.decode_tokens_by_ratio: Counter[float] = Counter()

    @property
    def intermediate_size(self) -> int:
        return int(self.down_proj.weight.shape[1])

    def set_ratio(self, ratio: float) -> None:
        ratio = float(ratio)
        if ratio not in self.supported_ratios:
            raise ValueError(f"Unsupported runtime ratio {ratio}; choose one of {self.supported_ratios}")
        self.active_ratio = ratio

    def reset_runtime_state(self) -> None:
        self.active_ratio = 0.0
        self.ranking = None
        self._mask_cache.clear()
        self.prefill_tokens = 0
        self.decode_tokens_by_ratio.clear()

    def _refresh_ranking(self, intermediate_states: torch.Tensor) -> None:
        self.ranking = rank_intermediate_neurons(intermediate_states.detach())
        self._mask_cache.clear()

    def keep_mask(self, ratio: float | None = None) -> torch.Tensor:
        ratio = self.active_ratio if ratio is None else float(ratio)
        if self.ranking is None:
            ranking = torch.arange(self.intermediate_size, device=self.down_proj.weight.device)
        else:
            ranking = self.ranking
        cached = self._mask_cache.get(ratio)
        if cached is None or cached.device != ranking.device:
            cached = keep_mask_from_ranking(ranking, ratio)
            self._mask_cache[ratio] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if x.shape[1] > 1 or self.ranking is None:
            self.prefill_tokens += int(x.shape[1])
            self._refresh_ranking(intermediate_states)
            return self.down_proj(intermediate_states)

        self.decode_tokens_by_ratio[self.active_ratio] += int(x.shape[1])
        if self.active_ratio <= 0.0:
            return self.down_proj(intermediate_states)
        mask = self.keep_mask().to(device=intermediate_states.device, dtype=intermediate_states.dtype)
        return self.down_proj(intermediate_states * mask)

    def runtime_summary(self) -> dict[str, Any]:
        decode_tokens = sum(self.decode_tokens_by_ratio.values())
        weighted_pruned = sum(ratio * tokens for ratio, tokens in self.decode_tokens_by_ratio.items())
        average_ratio = weighted_pruned / decode_tokens if decode_tokens else 0.0
        return {
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": decode_tokens,
            "decode_tokens_by_ratio": {f"{ratio:.2f}": count for ratio, count in sorted(self.decode_tokens_by_ratio.items())},
            "average_decode_pruning_ratio": average_ratio,
            "intermediate_size": self.intermediate_size,
        }


def get_runtime_mlp_layers(model: nn.Module) -> list[RuntimeMaskedQwen3MLP]:
    wrappers = []
    for layer in get_decoder_layers(model):
        if isinstance(getattr(layer, "mlp", None), RuntimeMaskedQwen3MLP):
            wrappers.append(layer.mlp)
    return wrappers


def set_runtime_mlp_ratio(model: nn.Module, ratio: float) -> None:
    wrappers = get_runtime_mlp_layers(model)
    if not wrappers:
        raise ValueError("Runtime RASP-Zero MLP adapter has not been applied")
    for wrapper in wrappers:
        wrapper.set_ratio(ratio)


def reset_runtime_mlp_state(model: nn.Module) -> None:
    for wrapper in get_runtime_mlp_layers(model):
        wrapper.reset_runtime_state()


def summarize_runtime_mlp(model: nn.Module) -> dict[str, Any]:
    wrappers = get_runtime_mlp_layers(model)
    layer_summaries = [wrapper.runtime_summary() for wrapper in wrappers]
    decode_tokens = sum(item["decode_tokens"] for item in layer_summaries)
    weighted_pruned = sum(item["average_decode_pruning_ratio"] * item["decode_tokens"] for item in layer_summaries)
    return {
        "backend": "logical_mask_v0",
        "real_speedup_claimed": False,
        "layers": len(wrappers),
        "average_decode_pruning_ratio": weighted_pruned / decode_tokens if decode_tokens else 0.0,
        "layer_summaries": layer_summaries,
    }


def apply_runtime_mlp_masking_qwen3(
    model: nn.Module,
    ratios: Iterable[float] = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40),
) -> nn.Module:
    layers = get_decoder_layers(model)
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, RuntimeMaskedQwen3MLP):
            continue
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj/act_fn")
        layer.mlp = RuntimeMaskedQwen3MLP(mlp, ratios)
    return model
