from __future__ import annotations

from collections import Counter, deque
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.hooks import get_decoder_layers

from .mask_bank import ratio_key, validate_mask_bank
from .protocol import STAGES
from .stage_risk import StageRiskController, stage_risk_features


def griffin_activation_score(intermediate_states: torch.Tensor) -> torch.Tensor:
    """GRIFFIN-style sequence score for FFN intermediate channels."""

    if intermediate_states.ndim != 3:
        raise ValueError(
            "Expected intermediate states with shape [batch, sequence, channels], "
            f"got {tuple(intermediate_states.shape)}"
        )
    if intermediate_states.shape[0] != 1:
        raise ValueError("Adaptive stage GRIFFIN runtime currently supports batch size 1")
    values = intermediate_states.detach().float()
    normalized = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return normalized.square().sum(dim=(0, 1)).sqrt()


def _zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-12)


def _keep_mask_from_scores(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError(f"Pruning ratio must be in [0, 1), got {ratio}")
    channels = int(scores.numel())
    keep = max(1, channels - int(round(channels * float(ratio))))
    indices = torch.topk(scores.float(), k=keep, largest=True).indices
    mask = torch.zeros(channels, dtype=torch.bool, device=scores.device)
    mask[indices] = True
    return mask


def _mask_pruning_ratio(mask: torch.Tensor) -> float:
    mask = mask.detach().bool()
    if mask.numel() == 0:
        return 0.0
    return 1.0 - float(mask.float().mean().item())


def _prune_mask_from_scores(
    prune_scores: torch.Tensor,
    ratio: float,
    *,
    protected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError(f"Pruning ratio must be in [0, 1), got {ratio}")
    scores = prune_scores.float()
    channels = int(scores.numel())
    prune = min(channels - 1, max(0, int(round(channels * float(ratio)))))
    mask = torch.ones(channels, dtype=torch.bool, device=scores.device)
    if prune == 0:
        return mask
    candidates = torch.ones(channels, dtype=torch.bool, device=scores.device)
    if protected_mask is not None:
        candidates &= ~protected_mask.to(device=scores.device, dtype=torch.bool)
    candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return mask
    prune = min(prune, int(candidate_indices.numel()))
    local_scores = scores[candidate_indices]
    prune_indices = candidate_indices[torch.topk(local_scores, k=prune, largest=True).indices]
    mask[prune_indices] = False
    return mask


def _topk_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Protected core fraction must be in [0, 1], got {fraction}")
    channels = int(scores.numel())
    keep = min(channels, max(0, int(round(channels * fraction))))
    mask = torch.zeros(channels, dtype=torch.bool, device=scores.device)
    if keep:
        mask[torch.topk(scores.float(), k=keep, largest=True).indices] = True
    return mask


def _topk_within_mask(scores: torch.Tensor, candidates: torch.Tensor, count: int) -> torch.Tensor:
    candidates = candidates.to(device=scores.device, dtype=torch.bool)
    output = torch.zeros_like(candidates, dtype=torch.bool)
    candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
    count = min(max(0, int(count)), int(candidate_indices.numel()))
    if count:
        local_scores = scores.float()[candidate_indices]
        output[candidate_indices[torch.topk(local_scores, k=count, largest=True).indices]] = True
    return output


def _safe_dynamic_score_mode(value: str) -> str:
    """Normalise legacy safe-dynamic names without changing old configs."""

    aliases = {
        "activation": "activation",
        "activation_keep": "activation",
        "protected_core_safe_prune": "activation",
        "current_safe": "activation",
            "output_aware": "output_aware",
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported safe-dynamic score mode: {value}") from exc


def _find_self_attention(layer: nn.Module) -> nn.Module:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("Could not locate self-attention module on decoder layer")


def _attention_num_heads(attn: nn.Module, hidden_size: int) -> int:
    for name in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
        value = getattr(attn, name, None)
        if value is not None:
            heads = int(value)
            if heads > 0 and hidden_size % heads == 0:
                return heads
    config = getattr(attn, "config", None)
    if config is not None:
        for name in ("num_attention_heads", "n_heads", "n_head"):
            value = getattr(config, name, None)
            if value is not None:
                heads = int(value)
                if heads > 0 and hidden_size % heads == 0:
                    return heads
    raise ValueError("Could not infer attention head count from module")


def _head_mask_from_scores(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError(f"Attention head pruning ratio must be in [0, 1), got {ratio}")
    heads = int(scores.numel())
    prune = min(heads - 1, max(0, int(round(heads * float(ratio)))))
    mask = torch.ones(heads, dtype=torch.bool, device=scores.device)
    if prune:
        prune_indices = torch.topk(scores.float(), k=prune, largest=False).indices
        mask[prune_indices] = False
    return mask


class StageMaskRuntime:
    def __init__(
        self,
        bank: dict[str, Any],
        policy: str,
        stage_ratios: dict[str, float],
        bias_compensation: bool = True,
        fallback_behavior: str = "dense_after_error",
    ) -> None:
        validate_mask_bank(bank)
        if policy not in bank["policies"]:
            raise ValueError(f"Unknown mask policy: {policy}")
        self.bank = bank
        self.policy = policy
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        self.bias_compensation = bool(bias_compensation)
        if fallback_behavior not in {"dense_after_error", "keep_masking_after_error"}:
            raise ValueError("Unsupported fallback behavior")
        self.fallback_behavior = str(fallback_behavior)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()
        self._actual_stage_ratio_cache: dict[str, float] = {}

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()
        self._actual_stage_ratio_cache.clear()

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            self.fallback_dense(f"unknown_stage:{stage}")
            return
        if self.fallback_reason is None:
            self.active_stage = stage

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        if self.fallback_behavior == "dense_after_error":
            self.active_stage = None

    def active_ratio(self) -> float:
        return self.stage_ratios[self.active_stage] if self.active_stage else 0.0

    def entry(self, layer_id: int) -> dict[str, Any] | None:
        if self.active_stage is None or self.active_ratio() <= 0.0:
            return None
        return self.bank["policies"][self.policy][self.active_stage][layer_id]

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def _mask_for_entry(self, entry: dict[str, Any], ratio: float) -> torch.Tensor:
        key = ratio_key(ratio)
        if key in entry["masks"]:
            return entry["masks"][key].bool()
        return _keep_mask_from_scores(entry["metric"], ratio)

    def actual_stage_ratio(self, stage: str) -> float:
        if stage not in STAGES:
            return 0.0
        cached = self._actual_stage_ratio_cache.get(stage)
        if cached is not None:
            return cached
        ratio = self.stage_ratios.get(stage, 0.0)
        if ratio <= 0.0:
            self._actual_stage_ratio_cache[stage] = 0.0
            return 0.0
        values = []
        for layer_id in self.bank["layers"]:
            entry = self.bank["policies"][self.policy][stage][int(layer_id)]
            values.append(_mask_pruning_ratio(self._mask_for_entry(entry, ratio)))
        actual = sum(values) / len(values) if values else 0.0
        self._actual_stage_ratio_cache[stage] = actual
        return actual

    def summary(self) -> dict[str, Any]:
        total = sum(self.tokens_by_stage.values())
        weighted = sum(
            self.stage_ratios.get(stage, 0.0) * count
            for stage, count in self.tokens_by_stage.items()
        )
        actual_weighted = sum(
            self.actual_stage_ratio(stage) * count
            for stage, count in self.tokens_by_stage.items()
        )
        return {
            "backend": "fixed_stage_mask_logical_v1",
            "real_speedup_claimed": False,
            "policy": self.policy,
            "stage_ratios": self.stage_ratios,
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "fallback_behavior": self.fallback_behavior,
            "tokens_by_stage": dict(self.tokens_by_stage),
            "theoretical_average_mlp_pruning_ratio": weighted / total if total else 0.0,
            "actual_average_mlp_pruning_ratio": actual_weighted / total if total else 0.0,
            "actual_pruning_accounting": "actual_mask_sparsity_token_weighted",
        }


class DenseStageRuntime:
    def __init__(self, policy: str = "dense") -> None:
        self.policy = str(policy)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()

    def set_stage(self, stage: str) -> None:
        if stage in STAGES:
            self.active_stage = stage
        else:
            self.fallback_dense(f"unknown_stage:{stage}")

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        self.active_stage = None

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "dense_no_mask_v1",
            "real_speedup_claimed": False,
            "policy": self.policy,
            "stage_ratios": {stage: 0.0 for stage in STAGES},
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "tokens_by_stage": dict(self.tokens_by_stage),
            "theoretical_average_mlp_pruning_ratio": 0.0,
            "actual_average_mlp_pruning_ratio": 0.0,
            "actual_pruning_accounting": "dense_no_mask",
        }


class StaticWeightPruningRuntime(DenseStageRuntime):
    def __init__(
        self,
        *,
        policy: str,
        backend: str,
        baseline_type: str,
        pruning_granularity: str,
        weight_sparsity_overall: float,
        extra_summary: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.backend = str(backend)
        self.baseline_type = str(baseline_type)
        self.pruning_granularity = str(pruning_granularity)
        self.weight_sparsity_overall = float(weight_sparsity_overall)
        self.extra_summary = dict(extra_summary or {})

    def summary(self) -> dict[str, Any]:
        output = super().summary()
        output.update(self.extra_summary)
        output.update(
            {
                "backend": self.backend,
                "baseline_type": self.baseline_type,
                "policy": self.policy,
                "pruning_granularity": self.pruning_granularity,
                "weight_sparsity_overall": self.weight_sparsity_overall,
                "theoretical_average_mlp_pruning_ratio": self.weight_sparsity_overall,
                "actual_average_mlp_pruning_ratio": self.weight_sparsity_overall,
                "actual_pruning_accounting": "static_weight_sparsity_over_target_linears",
            }
        )
        return output


class StaticMlpChannelPruningRuntime(DenseStageRuntime):
    def __init__(
        self,
        *,
        policy: str,
        backend: str,
        baseline_type: str,
        pruning_granularity: str,
        mlp_channel_pruning_ratio: float,
        extra_summary: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.backend = str(backend)
        self.baseline_type = str(baseline_type)
        self.pruning_granularity = str(pruning_granularity)
        self.mlp_channel_pruning_ratio = float(mlp_channel_pruning_ratio)
        self.extra_summary = dict(extra_summary or {})

    def summary(self) -> dict[str, Any]:
        output = super().summary()
        output.update(self.extra_summary)
        output.update(
            {
                "backend": self.backend,
                "baseline_type": self.baseline_type,
                "policy": self.policy,
                "pruning_granularity": self.pruning_granularity,
                "flap_actual_mlp_channel_pruning_ratio": self.mlp_channel_pruning_ratio,
                "theoretical_average_mlp_pruning_ratio": self.mlp_channel_pruning_ratio,
                "actual_average_mlp_pruning_ratio": self.mlp_channel_pruning_ratio,
                "actual_pruning_accounting": "static_mlp_channel_pruning_ratio",
            }
        )
        return output


class StaticLayerPruningRuntime(DenseStageRuntime):
    def __init__(
        self,
        *,
        policy: str,
        backend: str,
        baseline_type: str,
        pruning_granularity: str,
        total_layers: int,
        pruned_layers: list[int],
        extra_summary: dict[str, Any] | None = None,
        handles: list[Any] | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.backend = str(backend)
        self.baseline_type = str(baseline_type)
        self.pruning_granularity = str(pruning_granularity)
        self.total_layers = int(total_layers)
        self.pruned_layers = [int(layer_id) for layer_id in pruned_layers]
        self.extra_summary = dict(extra_summary or {})
        self._handles = list(handles or [])

    @property
    def layer_pruning_ratio(self) -> float:
        return len(self.pruned_layers) / self.total_layers if self.total_layers else 0.0

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> dict[str, Any]:
        output = super().summary()
        output.update(self.extra_summary)
        output.update(
            {
                "backend": self.backend,
                "baseline_type": self.baseline_type,
                "policy": self.policy,
                "pruning_granularity": self.pruning_granularity,
                "total_layers": self.total_layers,
                "pruned_layers": list(self.pruned_layers),
                "layer_pruning_ratio": self.layer_pruning_ratio,
                "density": 1.0 - self.layer_pruning_ratio,
                "theoretical_average_mlp_pruning_ratio": self.layer_pruning_ratio,
                "actual_average_mlp_pruning_ratio": self.layer_pruning_ratio,
                "actual_pruning_accounting": "static_decoder_layer_fraction",
            }
        )
        return output


class AlwaysOnStaticMaskRuntime(StageMaskRuntime):
    """Static global-mask baseline that records protocol fallback but keeps pruning.

    The regular fixed-stage runtime falls back to dense after an invalid stage
    sequence. That is correct for stage-conditioned methods, but it makes a
    static baseline's actual pruning depend on the stage protocol. This runtime
    keeps the same static mask active after fallback so it can serve as a strict
    actual-pruning matched baseline.
    """

    def __init__(
        self,
        bank: dict[str, Any],
        policy: str,
        stage_ratios: dict[str, float],
        bias_compensation: bool = True,
    ) -> None:
        super().__init__(
            bank,
            policy=policy,
            stage_ratios=stage_ratios,
            bias_compensation=bias_compensation,
        )
        self.runtime_policy = f"{policy}_always_on"

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        # Deliberately keep active_stage unchanged: static pruning should not
        # become dense merely because the generated stage tags are invalid.

    def summary(self) -> dict[str, Any]:
        output = super().summary()
        output["backend"] = "always_on_static_mask_logical_v1"
        output["policy"] = self.runtime_policy
        output["base_policy"] = self.policy
        output["fallback_behavior"] = "record_only_keep_masking"
        return output


class GriffinPromptRuntime:
    """Prompt-prompted dynamic FFN pruning baseline.

    The prompt/prefill pass is dense and only collects a per-layer GRIFFIN score.
    Subsequent single-token decoding uses the prompt-specific keep mask. Stage
    markers are recorded for diagnostics, but they do not control pruning.
    """

    def __init__(
        self,
        *,
        prune_ratio: float,
        selection_method: str = "topk",
    ) -> None:
        if not 0.0 <= float(prune_ratio) < 1.0:
            raise ValueError(f"GRIFFIN prune_ratio must be in [0, 1), got {prune_ratio}")
        if selection_method not in {"topk", "magnitude"}:
            raise ValueError(f"Unsupported GRIFFIN selection method: {selection_method}")
        self.prune_ratio = float(prune_ratio)
        self.selection_method = str(selection_method)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()
        self.dense_observation_tokens_by_stage: Counter[str] = Counter()
        self.masked_tokens_by_stage: Counter[str] = Counter()
        self.mask_refresh_count_by_stage: Counter[str] = Counter()
        self._prompt_score_sq: dict[int, torch.Tensor] = {}
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._keep_ratios_by_layer: dict[int, float] = {}
        self._actual_pruning_weighted_sum = 0.0
        self._actual_pruning_denominator = 0

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()
        self.dense_observation_tokens_by_stage.clear()
        self.masked_tokens_by_stage.clear()
        self.mask_refresh_count_by_stage.clear()
        self._prompt_score_sq.clear()
        self._mask_cache.clear()
        self._keep_ratios_by_layer.clear()
        self._actual_pruning_weighted_sum = 0.0
        self._actual_pruning_denominator = 0

    def set_stage(self, stage: str) -> None:
        if stage in STAGES:
            self.active_stage = stage
        else:
            self.fallback_dense(f"unknown_stage:{stage}")

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        # GRIFFIN is not stage-conditioned; protocol fallback is diagnostic only.

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def _record_actual_pruning(self, token_count: int, mask: torch.Tensor | None) -> None:
        ratio = 0.0 if mask is None else _mask_pruning_ratio(mask)
        count = int(token_count)
        self._actual_pruning_weighted_sum += ratio * count
        self._actual_pruning_denominator += count

    def _accumulate_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        score_sq = griffin_activation_score(intermediate).square()
        previous = self._prompt_score_sq.get(layer_id)
        self._prompt_score_sq[layer_id] = (
            score_sq if previous is None else previous.to(score_sq.device) + score_sq
        )
        self._mask_cache.pop(layer_id, None)

    def observe_or_mask(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor | None:
        token_count = int(intermediate.shape[1])
        if token_count > 1:
            self._accumulate_prompt(layer_id, intermediate)
            if layer_id == 0:
                self.dense_observation_tokens_by_stage["prompt"] += token_count
            self._record_actual_pruning(token_count, None)
            return None
        if self.prune_ratio <= 0.0:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[self.active_stage or "dense"] += token_count
            self._record_actual_pruning(token_count, None)
            return None
        mask = self.keep_mask(layer_id)
        if layer_id == 0:
            self.masked_tokens_by_stage[self.active_stage or "dense"] += token_count
        self._record_actual_pruning(token_count, mask)
        return mask

    def keep_mask(self, layer_id: int) -> torch.Tensor:
        cached = self._mask_cache.get(layer_id)
        if cached is not None:
            return cached
        score_sq = self._prompt_score_sq.get(layer_id)
        if score_sq is None:
            raise RuntimeError("GRIFFIN prompt score is unavailable before decode masking")
        mask = _keep_mask_from_scores(score_sq.float().sqrt(), self.prune_ratio)
        self._mask_cache[layer_id] = mask
        self._keep_ratios_by_layer[layer_id] = float(mask.float().mean().item())
        if layer_id == 0:
            self.mask_refresh_count_by_stage["prompt"] += 1
        return mask

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "griffin_prompt_logical_v1",
            "baseline_type": "prompt_prompted_dynamic",
            "real_speedup_claimed": False,
            "policy": "griffin_prompt",
            "score_mode": "prompt_activation_topk",
            "selection_method": self.selection_method,
            "prune_ratio": self.prune_ratio,
            "density": 1.0 - self.prune_ratio,
            "stage_ratios": {stage: self.prune_ratio for stage in STAGES},
            "tokens_by_stage": dict(self.tokens_by_stage),
            "dense_observation_tokens_by_stage": dict(self.dense_observation_tokens_by_stage),
            "masked_tokens_by_stage": dict(self.masked_tokens_by_stage),
            "mask_refresh_count_by_stage": dict(self.mask_refresh_count_by_stage),
            "prompt_dense_tokens": int(self.dense_observation_tokens_by_stage.get("prompt", 0)),
            "decode_masked_tokens": int(sum(self.masked_tokens_by_stage.values())),
            "keep_ratios_by_layer": {
                str(layer_id): ratio for layer_id, ratio in sorted(self._keep_ratios_by_layer.items())
            },
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "fallback_behavior": "record_only_keep_prompt_masking",
            "theoretical_average_mlp_pruning_ratio": (
                self._actual_pruning_weighted_sum / self._actual_pruning_denominator
                if self._actual_pruning_denominator
                else 0.0
            ),
            "actual_average_mlp_pruning_ratio": (
                self._actual_pruning_weighted_sum / self._actual_pruning_denominator
                if self._actual_pruning_denominator
                else 0.0
            ),
            "actual_pruning_accounting": "prompt_dense_and_decode_mask_layer_token_weighted",
        }


class FixedStageMaskedQwen3MLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, layer_id: int, runtime: StageMaskRuntime) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.layer_id = int(layer_id)
        self.runtime = runtime
        self._runtime_mask_cache: dict[tuple[str, str], torch.Tensor] = {}

    @property
    def intermediate_size(self) -> int:
        return int(self.down_proj.weight.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        entry = self.runtime.entry(self.layer_id)
        if entry is None:
            return self.down_proj(intermediate)
        ratio = self.runtime.active_ratio()
        key = ratio_key(ratio)
        if key in entry["masks"]:
            mask_bool = entry["masks"][key].to(device=intermediate.device, dtype=torch.bool)
        else:
            cache_key = (key, str(intermediate.device))
            mask_bool = self._runtime_mask_cache.get(cache_key)
            if mask_bool is None:
                mask_bool = _keep_mask_from_scores(
                    entry["metric"].to(device=intermediate.device),
                    ratio,
                )
                self._runtime_mask_cache[cache_key] = mask_bool
        mask = mask_bool.to(device=intermediate.device, dtype=intermediate.dtype)
        output = self.down_proj(intermediate * mask)
        if self.runtime.bias_compensation:
            removed_mean = entry["mean"].to(
                device=intermediate.device, dtype=intermediate.dtype
            ) * (1.0 - mask)
            compensation = removed_mean @ self.down_proj.weight.t()
            output = output + compensation
        return output


def apply_fixed_stage_masking_qwen3(model: nn.Module, runtime: StageMaskRuntime) -> nn.Module:
    layers = get_decoder_layers(model)
    if len(layers) != len(runtime.bank["layers"]):
        raise ValueError("Mask bank layer count does not match model")
    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FixedStageMaskedQwen3MLP):
            if mlp.runtime is not runtime:
                mlp.runtime = runtime
            continue
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected gated MLP with gate_proj/up_proj/down_proj/act_fn")
        layer.mlp = FixedStageMaskedQwen3MLP(mlp, layer_id, runtime)
    return model


class GriffinPromptQwen3MLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, layer_id: int, runtime: GriffinPromptRuntime) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.layer_id = int(layer_id)
        self.runtime = runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        mask = self.runtime.observe_or_mask(self.layer_id, intermediate)
        if mask is None:
            return self.down_proj(intermediate)
        mask = mask.to(device=intermediate.device, dtype=intermediate.dtype)
        return self.down_proj(intermediate * mask)


def apply_griffin_prompt_qwen3(model: nn.Module, runtime: GriffinPromptRuntime) -> nn.Module:
    layers = get_decoder_layers(model)
    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, GriffinPromptQwen3MLP):
            if mlp.runtime is not runtime:
                mlp.runtime = runtime
            continue
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected gated MLP with gate_proj/up_proj/down_proj/act_fn")
        layer.mlp = GriffinPromptQwen3MLP(mlp, layer_id, runtime)
    return model


class AdaptiveStageGriffinRuntime:
    def __init__(
        self,
        bank: dict[str, Any],
        *,
        stage_ratios: dict[str, float],
        alpha: float = 0.7,
        warmup_tokens: dict[str, int] | None = None,
        bias_compensation: bool = True,
        prior_policy: str = "stage_specific",
        fallback_behavior: str = "dense_after_error",
    ) -> None:
        validate_mask_bank(bank)
        if prior_policy not in bank["policies"]:
            raise ValueError(f"Unknown calibration prior policy: {prior_policy}")
        self.bank = bank
        self.prior_policy = prior_policy
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        warmup_tokens = warmup_tokens or {}
        self.warmup_tokens = {stage: int(warmup_tokens.get(stage, 0)) for stage in STAGES}
        self.bias_compensation = bool(bias_compensation)
        if fallback_behavior not in {"dense_after_error", "keep_masking_after_error"}:
            raise ValueError("Unsupported fallback behavior")
        self.fallback_behavior = str(fallback_behavior)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()
        self.dense_observation_tokens_by_stage: Counter[str] = Counter()
        self.masked_tokens_by_stage: Counter[str] = Counter()
        self.mask_refresh_count_by_stage: Counter[str] = Counter()
        self._prompt_scores: dict[int, torch.Tensor] = {}
        self._stage_scores: dict[str, dict[int, torch.Tensor]] = {stage: {} for stage in STAGES}
        self._stage_observed_tokens: Counter[str] = Counter()
        self._mask_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._current_single_stage: str | None = None
        self._current_single_observe = False

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()
        self.dense_observation_tokens_by_stage.clear()
        self.masked_tokens_by_stage.clear()
        self.mask_refresh_count_by_stage.clear()
        self._prompt_scores.clear()
        self._stage_scores = {stage: {} for stage in STAGES}
        self._stage_observed_tokens.clear()
        self._mask_cache.clear()
        self._current_single_stage = None
        self._current_single_observe = False

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            self.fallback_dense(f"unknown_stage:{stage}")
            return
        if self.fallback_reason is None:
            self.active_stage = stage
            self._current_single_stage = None
            self._current_single_observe = False

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        if self.fallback_behavior == "dense_after_error":
            self.active_stage = None
            self._current_single_stage = None
            self._current_single_observe = False

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def _accumulate(
        self,
        storage: dict[int, torch.Tensor],
        layer_id: int,
        intermediate: torch.Tensor,
    ) -> None:
        score_sq = griffin_activation_score(intermediate).square()
        previous = storage.get(layer_id)
        storage[layer_id] = score_sq if previous is None else previous.to(score_sq.device) + score_sq
        self._mask_cache = {
            key: value for key, value in self._mask_cache.items() if key[1] != layer_id
        }

    def observe_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        self._accumulate(self._prompt_scores, layer_id, intermediate)

    def _single_token_mode(self, layer_id: int, token_count: int) -> tuple[str | None, bool]:
        if layer_id == 0:
            stage = self.active_stage
            observe = False
            if stage in STAGES and self.stage_ratios[stage] > 0.0:
                observe = self._stage_observed_tokens[stage] < self.warmup_tokens[stage]
                if observe:
                    self._stage_observed_tokens[stage] += int(token_count)
            self._current_single_stage = stage
            self._current_single_observe = observe
        return self._current_single_stage, self._current_single_observe

    def observe_or_mask(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor | None:
        token_count = int(intermediate.shape[1])
        if token_count > 1:
            self.observe_prompt(layer_id, intermediate)
            return None
        stage, observe = self._single_token_mode(layer_id, token_count)
        if (self.fallback_reason is not None and self.fallback_behavior == "dense_after_error") or stage not in STAGES:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage["dense"] += token_count
            return None
        ratio = self.stage_ratios[stage]
        if ratio <= 0.0:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            return None
        if observe:
            self._accumulate(self._stage_scores[stage], layer_id, intermediate)
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            return None
        if layer_id == 0:
            self.masked_tokens_by_stage[stage] += token_count
        return self.keep_mask(stage, layer_id)

    def keep_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        key = (stage, layer_id)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        entry = self.bank["policies"][self.prior_policy][stage][layer_id]
        prior = entry["metric"].float()
        runtime_score = self._stage_scores[stage].get(layer_id)
        if runtime_score is None:
            runtime_score = self._prompt_scores.get(layer_id)
        if runtime_score is None:
            runtime_score = torch.ones_like(prior)
        runtime_score = runtime_score.to(device=prior.device, dtype=torch.float32).sqrt()
        score = self.alpha * _zscore(runtime_score) + (1.0 - self.alpha) * _zscore(prior)
        mask = _keep_mask_from_scores(score, self.stage_ratios[stage])
        self._mask_cache[key] = mask
        if layer_id == 0:
            self.mask_refresh_count_by_stage[stage] += 1
        return mask

    def entry(self, stage: str, layer_id: int) -> dict[str, Any]:
        return self.bank["policies"][self.prior_policy][stage][layer_id]

    def summary(self) -> dict[str, Any]:
        total = sum(self.dense_observation_tokens_by_stage.values()) + sum(
            self.masked_tokens_by_stage.values()
        )
        weighted = sum(
            self.stage_ratios.get(stage, 0.0) * count
            for stage, count in self.masked_tokens_by_stage.items()
        )
        return {
            "backend": "calibrated_stage_adaptive_griffin_logical_v1",
            "real_speedup_claimed": False,
            "policy": "calibrated_stage_adaptive_griffin",
            "alpha": self.alpha,
            "warmup_tokens": self.warmup_tokens,
            "stage_ratios": self.stage_ratios,
            "tokens_by_stage": dict(self.tokens_by_stage),
            "dense_observation_tokens_by_stage": dict(self.dense_observation_tokens_by_stage),
            "masked_tokens_by_stage": dict(self.masked_tokens_by_stage),
            "mask_refresh_count_by_stage": dict(self.mask_refresh_count_by_stage),
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "fallback_behavior": self.fallback_behavior,
            "theoretical_average_mlp_pruning_ratio": weighted / total if total else 0.0,
        }


class SafeDynamicStageGriffinRuntime:
    def __init__(
        self,
        bank: dict[str, Any],
        *,
        stage_ratios: dict[str, float],
        runtime_weight: float = 0.4,
        prior_weight: float = 0.6,
        warmup_tokens: dict[str, int] | None = None,
        protected_core_ratios: dict[str, float] | None = None,
        refresh_intervals: dict[str, int] | None = None,
        window_tokens: dict[str, int] | None = None,
        bias_compensation: bool = True,
        prior_policy: str = "stage_specific",
        fallback_behavior: str = "dense_after_error",
        score_mode: str = "activation",
        max_mask_swap_fraction: float = 1.0,
        stage_budget_controller: dict[str, Any] | None = None,
        attention_head_pruning: dict[str, Any] | None = None,
        multi_structure_budget_controller: dict[str, Any] | None = None,
    ) -> None:
        validate_mask_bank(bank)
        if prior_policy not in bank["policies"]:
            raise ValueError(f"Unknown calibration prior policy: {prior_policy}")
        self.bank = bank
        self.prior_policy = prior_policy
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        self.runtime_weight = float(runtime_weight)
        self.prior_weight = float(prior_weight)
        if self.runtime_weight < 0.0 or self.prior_weight < 0.0:
            raise ValueError("runtime_weight and prior_weight must be non-negative")
        if self.runtime_weight + self.prior_weight <= 0.0:
            raise ValueError("At least one safe-dynamic score weight must be positive")
        self.score_mode = _safe_dynamic_score_mode(score_mode)
        self.max_mask_swap_fraction = float(max_mask_swap_fraction)
        if not 0.0 <= self.max_mask_swap_fraction <= 1.0:
            raise ValueError("max_mask_swap_fraction must be in [0, 1]")
        budget_cfg = dict(stage_budget_controller or {})
        self.stage_budget_controller_enabled = bool(budget_cfg.get("enabled", False))
        self.stage_budget_controller = self._prepare_stage_budget_controller(budget_cfg)
        self.attention_head_pruning = self._prepare_attention_head_pruning(
            dict(attention_head_pruning or {})
        )
        self.attention_head_pruning_enabled = bool(self.attention_head_pruning["enabled"])
        self.multi_structure_budget_controller = self._prepare_multi_structure_budget_controller(
            dict(multi_structure_budget_controller or {})
        )
        self.multi_structure_budget_controller_enabled = bool(
            self.multi_structure_budget_controller["enabled"]
        )
        warmup_tokens = warmup_tokens or {}
        protected_core_ratios = protected_core_ratios or {}
        refresh_intervals = refresh_intervals or {}
        window_tokens = window_tokens or {}
        self.warmup_tokens = {stage: int(warmup_tokens.get(stage, 0)) for stage in STAGES}
        self.protected_core_ratios = {
            stage: float(protected_core_ratios.get(stage, 0.0)) for stage in STAGES
        }
        self.refresh_intervals = {
            stage: int(refresh_intervals.get(stage, 0)) for stage in STAGES
        }
        self.window_tokens = {
            stage: max(1, int(window_tokens.get(stage, self.refresh_intervals[stage] or 1)))
            for stage in STAGES
        }
        for stage, value in self.protected_core_ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid protected core ratio for {stage}: {value}")
        self.bias_compensation = bool(bias_compensation)
        if fallback_behavior not in {"dense_after_error", "keep_masking_after_error"}:
            raise ValueError("Unsupported fallback behavior")
        self.fallback_behavior = str(fallback_behavior)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()
        self.dense_observation_tokens_by_stage: Counter[str] = Counter()
        self.masked_tokens_by_stage: Counter[str] = Counter()
        self.mask_refresh_count_by_stage: Counter[str] = Counter()
        self._prompt_scores: dict[int, torch.Tensor] = {}
        self._recent_scores: dict[str, dict[int, deque[torch.Tensor]]] = {stage: {} for stage in STAGES}
        self._recent_sums: dict[str, dict[int, torch.Tensor]] = {stage: {} for stage in STAGES}
        self._stage_observed_tokens: Counter[str] = Counter()
        self._stage_tokens_since_refresh: Counter[str] = Counter()
        self._mask_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._current_single_stage: str | None = None
        self._current_single_observe = False
        self._actual_pruning_weighted_sum = 0.0
        self._actual_pruning_denominator = 0
        self._actual_pruning_weighted_sum_by_stage: Counter[str] = Counter()
        self._actual_pruning_denominator_by_stage: Counter[str] = Counter()
        self._output_norms: dict[int, torch.Tensor] = {}
        self._output_norm_hash: str | None = None
        self._previous_masks: dict[int, torch.Tensor] = {}
        self._mask_swap_pairs_by_stage_layer: Counter[str] = Counter()
        self._mask_swap_candidates_by_stage_layer: Counter[str] = Counter()
        self._mask_jaccard_sum_by_stage_layer: Counter[str] = Counter()
        self._mask_jaccard_count_by_stage_layer: Counter[str] = Counter()
        self._budget_selected_ratios: dict[str, float] = {
            stage: self._initial_budget_ratio(stage) for stage in STAGES
        }
        self._budget_tokens_since_decision: Counter[str] = Counter()
        self._budget_decision_log: list[dict[str, Any]] = []
        self._budget_ratio_tokens: Counter[str] = Counter()
        self._budget_base_risks: dict[str, list[float]] = {stage: [] for stage in STAGES}
        self._budget_margin_risks: dict[str, list[float]] = {stage: [] for stage in STAGES}
        self._budget_volatility_risks: dict[str, list[float]] = {stage: [] for stage in STAGES}
        self._budget_debts: list[float] = []
        self._budget_previous_scores: dict[str, torch.Tensor] = {}
        self._attention_num_heads_by_layer: dict[int, int] = {}
        self._attention_head_dim_by_layer: dict[int, int] = {}
        self._attention_o_proj_norms: dict[int, torch.Tensor] = {}
        self._attention_o_proj_norm_hash: str | None = None
        self._attention_prompt_scores: dict[int, torch.Tensor] = {}
        self._attention_recent_scores: dict[str, dict[int, deque[torch.Tensor]]] = {
            stage: {} for stage in STAGES
        }
        self._attention_recent_sums: dict[str, dict[int, torch.Tensor]] = {
            stage: {} for stage in STAGES
        }
        self._attention_mask_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._attention_actual_weighted_sum = 0.0
        self._attention_actual_denominator = 0
        self._attention_actual_weighted_sum_by_stage: Counter[str] = Counter()
        self._attention_actual_denominator_by_stage: Counter[str] = Counter()
        self._attention_ratio_tokens: Counter[str] = Counter()
        self._attention_selected_ratios: dict[str, float] = {
            stage: float(self.attention_head_pruning["stage_ratios"].get(stage, 0.0))
            for stage in STAGES
        }
        self._attention_tokens_since_decision: Counter[str] = Counter()
        self._attention_decision_log: list[dict[str, Any]] = []
        self._attention_budget_debts: list[float] = []
        self._attention_previous_scores: dict[str, torch.Tensor] = {}
        self._multi_selected_mlp_ratios: dict[str, float] = {
            stage: self._initial_budget_ratio(stage) for stage in STAGES
        }
        self._multi_selected_attention_ratios: dict[str, float] = {
            stage: float(self.attention_head_pruning["stage_ratios"].get(stage, 0.0))
            for stage in STAGES
        }
        self._multi_tokens_since_decision: Counter[str] = Counter()
        self._multi_decision_log: list[dict[str, Any]] = []

    def _prepare_stage_budget_controller(self, cfg: dict[str, Any]) -> dict[str, Any]:
        action_ratios = tuple(
            sorted({float(value) for value in cfg.get("action_ratios", [0.0, 0.2, 0.3, 0.4])})
        )
        if not action_ratios or action_ratios[0] != 0.0:
            raise ValueError("stage_budget_controller.action_ratios must include dense ratio 0.0")
        if any(value < 0.0 or value >= 1.0 for value in action_ratios):
            raise ValueError("Invalid stage budget action ratio")
        target = float(cfg.get("target_actual_pruning", 0.34))
        if not 0.0 <= target < 1.0:
            raise ValueError("Invalid stage budget pruning target")
        raw_bounds = cfg.get("stage_ratio_bounds", {})
        default_bounds = {
            "setup": (0.20, 0.42),
            "reasoning": (0.20, 0.38),
            "verify": (0.10, 0.34),
            "final": (0.00, 0.20),
        }
        bounds: dict[str, tuple[float, float]] = {}
        for stage in STAGES:
            values = raw_bounds.get(stage, default_bounds[stage])
            if isinstance(values, dict):
                low = float(values.get("floor", default_bounds[stage][0]))
                high = float(values.get("cap", default_bounds[stage][1]))
            else:
                low, high = [float(value) for value in values]
            if low < 0.0 or high >= 1.0 or low > high:
                raise ValueError(f"Invalid stage budget bounds for {stage}: {values}")
            bounds[stage] = (low, high)
        stage_risk_bias = {
            "setup": -0.05,
            "reasoning": 0.00,
            "verify": 0.08,
            "final": 0.15,
        }
        stage_risk_bias.update({stage: float(value) for stage, value in cfg.get("stage_risk_bias", {}).items()})
        ratio_selection_mode = str(cfg.get("ratio_selection_mode", "nominal"))
        if ratio_selection_mode not in {"nominal", "estimated_actual"}:
            raise ValueError(
                "stage_budget_controller.ratio_selection_mode must be "
                "'nominal' or 'estimated_actual'"
            )
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "target_actual_pruning": target,
            "decision_window_tokens": max(1, int(cfg.get("decision_window_tokens", 128))),
            "action_ratios": action_ratios,
            "stage_ratio_bounds": bounds,
            "stage_risk_bias": {stage: float(stage_risk_bias.get(stage, 0.0)) for stage in STAGES},
            "risk_dense_threshold": float(cfg.get("risk_dense_threshold", 0.85)),
            "risk_reduce_threshold": float(cfg.get("risk_reduce_threshold", 0.62)),
            "risk_safe_threshold": float(cfg.get("risk_safe_threshold", 0.35)),
            "budget_debt_gain": float(cfg.get("budget_debt_gain", 0.75)),
            "budget_catchup_threshold": float(cfg.get("budget_catchup_threshold", 0.015)),
            "risk_ratio_penalty": float(cfg.get("risk_ratio_penalty", 0.10)),
            "ratio_selection_mode": ratio_selection_mode,
        }

    def _prepare_attention_head_pruning(self, cfg: dict[str, Any]) -> dict[str, Any]:
        enabled = bool(cfg.get("enabled", False))
        stage_ratios = cfg.get("stage_ratios", {})
        prepared_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        for stage, ratio in prepared_ratios.items():
            if not 0.0 <= ratio < 1.0:
                raise ValueError(f"Invalid attention head ratio for {stage}: {ratio}")
        action_ratios = tuple(
            sorted({float(value) for value in cfg.get("action_ratios", [0.0, 0.0625, 0.125])})
        )
        if not action_ratios or action_ratios[0] != 0.0:
            raise ValueError("attention_head_pruning.action_ratios must include 0.0")
        if any(value < 0.0 or value >= 1.0 for value in action_ratios):
            raise ValueError("Invalid attention head action ratio")
        raw_bounds = cfg.get("stage_ratio_bounds", {})
        default_bounds = {
            "setup": (0.0, 0.125),
            "reasoning": (0.0, 0.125),
            "verify": (0.0, 0.0625),
            "final": (0.0, 0.0),
        }
        bounds: dict[str, tuple[float, float]] = {}
        for stage in STAGES:
            values = raw_bounds.get(stage, default_bounds[stage])
            low, high = [float(value) for value in values]
            if low < 0.0 or high >= 1.0 or low > high:
                raise ValueError(f"Invalid attention head bounds for {stage}: {values}")
            bounds[stage] = (low, high)
        score_mode = str(cfg.get("score_mode", "output_aware"))
        if score_mode != "output_aware":
            raise ValueError("attention_head_pruning.score_mode currently supports output_aware only")
        return {
            "enabled": enabled,
            "score_mode": score_mode,
            "stage_ratios": prepared_ratios,
            "action_ratios": action_ratios,
            "stage_ratio_bounds": bounds,
            "target_attention_pruning": float(cfg.get("target_attention_pruning", 0.10)),
            "decision_window_tokens": max(1, int(cfg.get("decision_window_tokens", 128))),
            "budget_debt_gain": float(cfg.get("budget_debt_gain", 0.75)),
            "budget_catchup_threshold": float(cfg.get("budget_catchup_threshold", 0.01)),
            "risk_ratio_penalty": float(cfg.get("risk_ratio_penalty", 0.02)),
            "dynamic_budget": bool(cfg.get("dynamic_budget", False)),
        }

    def _prepare_multi_structure_budget_controller(self, cfg: dict[str, Any]) -> dict[str, Any]:
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return {"enabled": False}
        mlp_action_ratios = tuple(
            sorted({float(value) for value in cfg.get("mlp_action_ratios", self.stage_budget_controller["action_ratios"])})
        )
        attention_action_ratios = tuple(
            sorted({float(value) for value in cfg.get("attention_action_ratios", self.attention_head_pruning["action_ratios"])})
        )
        if not mlp_action_ratios or mlp_action_ratios[0] != 0.0:
            raise ValueError("multi_structure_budget_controller.mlp_action_ratios must include 0.0")
        if not attention_action_ratios or attention_action_ratios[0] != 0.0:
            raise ValueError("multi_structure_budget_controller.attention_action_ratios must include 0.0")

        def parse_bounds(key: str, defaults: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
            raw = cfg.get(key, {})
            out: dict[str, tuple[float, float]] = {}
            for stage in STAGES:
                low, high = [float(value) for value in raw.get(stage, defaults[stage])]
                if low < 0.0 or high >= 1.0 or low > high:
                    raise ValueError(f"Invalid {key} for {stage}: {raw.get(stage)}")
                out[stage] = (low, high)
            return out

        stage_attention_penalty = {
            "setup": 0.0,
            "reasoning": 0.04,
            "verify": 0.08,
            "final": 0.20,
        }
        stage_attention_penalty.update(
            {stage: float(value) for stage, value in cfg.get("stage_attention_penalty", {}).items()}
        )
        return {
            "enabled": True,
            "decision_window_tokens": max(1, int(cfg.get("decision_window_tokens", 128))),
            "mlp_target_actual_pruning": float(cfg.get("mlp_target_actual_pruning", 0.315)),
            "attention_target_actual_pruning": float(cfg.get("attention_target_actual_pruning", 0.060)),
            "mlp_action_ratios": mlp_action_ratios,
            "attention_action_ratios": attention_action_ratios,
            "mlp_stage_bounds": parse_bounds(
                "mlp_stage_bounds",
                {stage: self.stage_budget_controller["stage_ratio_bounds"][stage] for stage in STAGES},
            ),
            "attention_stage_bounds": parse_bounds(
                "attention_stage_bounds",
                {stage: self.attention_head_pruning["stage_ratio_bounds"][stage] for stage in STAGES},
            ),
            "stage_attention_penalty": {
                stage: float(stage_attention_penalty.get(stage, 0.0)) for stage in STAGES
            },
            "budget_debt_gain": float(cfg.get("budget_debt_gain", 0.75)),
            "risk_ratio_penalty": float(cfg.get("risk_ratio_penalty", 0.02)),
        }

    def _initial_budget_ratio(self, stage: str) -> float:
        if not self.stage_budget_controller_enabled:
            return self.stage_ratios.get(stage, 0.0)
        floor, cap = self.stage_budget_controller["stage_ratio_bounds"][stage]
        candidates = self._stage_budget_candidates(stage, include_dense=False)
        if not candidates:
            return 0.0
        desired = min(cap, max(floor, float(self.stage_ratios.get(stage, 0.0))))
        return min(candidates, key=lambda value: (abs(value - desired), -value))

    def _stage_budget_candidates(self, stage: str, *, include_dense: bool = True) -> list[float]:
        if not self.stage_budget_controller_enabled:
            return [self.stage_ratios.get(stage, 0.0)]
        floor, cap = self.stage_budget_controller["stage_ratio_bounds"][stage]
        output = [
            ratio for ratio in self.stage_budget_controller["action_ratios"]
            if (ratio == 0.0 and include_dense) or (floor - 1e-12 <= ratio <= cap + 1e-12)
        ]
        return sorted(set(float(value) for value in output))

    def _active_stage_ratio(self, stage: str) -> float:
        if self.multi_structure_budget_controller_enabled:
            return float(self._multi_selected_mlp_ratios.get(stage, 0.0))
        if self.stage_budget_controller_enabled:
            return float(self._budget_selected_ratios.get(stage, 0.0))
        return float(self.stage_ratios.get(stage, 0.0))

    @property
    def alpha(self) -> float:
        total = self.runtime_weight + self.prior_weight
        return self.runtime_weight / total

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()
        self.dense_observation_tokens_by_stage.clear()
        self.masked_tokens_by_stage.clear()
        self.mask_refresh_count_by_stage.clear()
        self._prompt_scores.clear()
        self._recent_scores = {stage: {} for stage in STAGES}
        self._recent_sums = {stage: {} for stage in STAGES}
        self._stage_observed_tokens.clear()
        self._stage_tokens_since_refresh.clear()
        self._mask_cache.clear()
        self._current_single_stage = None
        self._current_single_observe = False
        self._actual_pruning_weighted_sum = 0.0
        self._actual_pruning_denominator = 0
        self._actual_pruning_weighted_sum_by_stage.clear()
        self._actual_pruning_denominator_by_stage.clear()
        self._previous_masks.clear()
        self._mask_swap_pairs_by_stage_layer.clear()
        self._mask_swap_candidates_by_stage_layer.clear()
        self._mask_jaccard_sum_by_stage_layer.clear()
        self._mask_jaccard_count_by_stage_layer.clear()
        self._budget_selected_ratios = {
            stage: self._initial_budget_ratio(stage) for stage in STAGES
        }
        self._budget_tokens_since_decision.clear()
        self._budget_decision_log.clear()
        self._budget_ratio_tokens.clear()
        self._budget_base_risks = {stage: [] for stage in STAGES}
        self._budget_margin_risks = {stage: [] for stage in STAGES}
        self._budget_volatility_risks = {stage: [] for stage in STAGES}
        self._budget_debts.clear()
        self._budget_previous_scores.clear()
        self._attention_prompt_scores.clear()
        self._attention_recent_scores = {stage: {} for stage in STAGES}
        self._attention_recent_sums = {stage: {} for stage in STAGES}
        self._attention_mask_cache.clear()
        self._attention_actual_weighted_sum = 0.0
        self._attention_actual_denominator = 0
        self._attention_actual_weighted_sum_by_stage.clear()
        self._attention_actual_denominator_by_stage.clear()
        self._attention_ratio_tokens.clear()
        self._attention_selected_ratios = {
            stage: float(self.attention_head_pruning["stage_ratios"].get(stage, 0.0))
            for stage in STAGES
        }
        self._attention_tokens_since_decision.clear()
        self._attention_decision_log.clear()
        self._attention_budget_debts.clear()
        self._attention_previous_scores.clear()
        self._multi_selected_mlp_ratios = {
            stage: self._initial_budget_ratio(stage) for stage in STAGES
        }
        self._multi_selected_attention_ratios = {
            stage: float(self.attention_head_pruning["stage_ratios"].get(stage, 0.0))
            for stage in STAGES
        }
        self._multi_tokens_since_decision.clear()
        self._multi_decision_log.clear()

    def set_stage(self, stage: str) -> None:
        previous = self.active_stage
        if stage not in STAGES:
            self.fallback_dense(f"unknown_stage:{stage}")
            return
        if self.fallback_reason is None:
            self.active_stage = stage
            self._current_single_stage = None
            self._current_single_observe = False
            if previous != stage and self.stage_budget_controller_enabled:
                self._budget_tokens_since_decision[stage] = int(
                    self.stage_budget_controller["decision_window_tokens"]
                )
            if previous != stage and self.attention_head_pruning_enabled:
                self._attention_tokens_since_decision[stage] = int(
                    self.attention_head_pruning["decision_window_tokens"]
                )
                self._attention_clear_stage_cache(stage)
            if previous != stage and self.multi_structure_budget_controller_enabled:
                self._multi_tokens_since_decision[stage] = int(
                    self.multi_structure_budget_controller["decision_window_tokens"]
                )
            if previous != stage and self.max_mask_swap_fraction < 1.0:
                self._clear_stage_cache(stage)

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        if self.fallback_behavior == "dense_after_error":
            self.active_stage = None
            self._current_single_stage = None
            self._current_single_observe = False

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def _accumulate_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        score_sq = self._runtime_channel_score(layer_id, intermediate)
        previous = self._prompt_scores.get(layer_id)
        self._prompt_scores[layer_id] = score_sq if previous is None else previous.to(score_sq.device) + score_sq

    def _append_recent(self, stage: str, layer_id: int, intermediate: torch.Tensor) -> None:
        score_sq = self._runtime_channel_score(layer_id, intermediate)
        by_layer = self._recent_scores[stage]
        queue = by_layer.get(layer_id)
        if queue is None:
            queue = deque()
            by_layer[layer_id] = queue
        maxlen = self.window_tokens[stage]
        current_sum = self._recent_sums[stage].get(layer_id)
        if current_sum is None:
            current_sum = torch.zeros_like(score_sq)
        else:
            current_sum = current_sum.to(score_sq.device)
        while len(queue) >= maxlen:
            current_sum = current_sum - queue.popleft().to(score_sq.device)
        queue.append(score_sq.detach().cpu())
        self._recent_sums[stage][layer_id] = current_sum + score_sq

    def set_output_norms(self, output_norms: dict[int, torch.Tensor]) -> None:
        """Attach channel norms read from the frozen model's down projections."""

        expected_layers = {int(layer_id) for layer_id in self.bank["layers"]}
        if set(output_norms) != expected_layers:
            raise ValueError("Output norms must cover exactly the mask-bank layers")
        digest = hashlib.sha256()
        prepared: dict[int, torch.Tensor] = {}
        for layer_id in sorted(expected_layers):
            norm = output_norms[layer_id].detach().float().contiguous().cpu()
            expected_width = int(
                self.bank["policies"][self.prior_policy][STAGES[0]][layer_id]["metric"].numel()
            )
            if int(norm.numel()) != expected_width:
                raise ValueError(f"Output-norm width mismatch for layer {layer_id}")
            digest.update(str(layer_id).encode("ascii"))
            digest.update(norm.numpy().tobytes())
            prepared[layer_id] = norm
        self._output_norms = prepared
        self._output_norm_hash = digest.hexdigest()

    def _runtime_channel_score(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "activation":
            return griffin_activation_score(intermediate).square()
        output_norm = self._output_norms.get(layer_id)
        if output_norm is None:
            raise RuntimeError(
                "output_aware safe-dynamic scoring requires down_proj norms from the frozen model"
            )
        activation_energy = intermediate.detach().float().square().mean(dim=(0, 1))
        return activation_energy * output_norm.to(device=activation_energy.device, dtype=torch.float32)

    def observe_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        self._accumulate_prompt(layer_id, intermediate)

    def _clear_stage_cache(self, stage: str) -> None:
        self._mask_cache = {key: value for key, value in self._mask_cache.items() if key[0] != stage}

    def register_attention_layer(
        self,
        layer_id: int,
        *,
        num_heads: int,
        head_dim: int,
        o_proj_weight: torch.Tensor,
    ) -> None:
        self._attention_num_heads_by_layer[int(layer_id)] = int(num_heads)
        self._attention_head_dim_by_layer[int(layer_id)] = int(head_dim)
        weight = o_proj_weight.detach().float().cpu()
        expected_hidden = int(num_heads) * int(head_dim)
        if int(weight.shape[1]) != expected_hidden:
            raise ValueError(
                f"o_proj input width mismatch for layer {layer_id}: "
                f"{weight.shape[1]} vs {expected_hidden}"
            )
        norms = weight.square().sum(dim=0).reshape(int(num_heads), int(head_dim)).sum(dim=1)
        self._attention_o_proj_norms[int(layer_id)] = norms.contiguous()
        digest = hashlib.sha256()
        for lid in sorted(self._attention_o_proj_norms):
            digest.update(str(lid).encode("ascii"))
            digest.update(self._attention_o_proj_norms[lid].numpy().tobytes())
        self._attention_o_proj_norm_hash = digest.hexdigest()

    def _attention_clear_stage_cache(self, stage: str) -> None:
        self._attention_mask_cache = {
            key: value for key, value in self._attention_mask_cache.items() if key[0] != stage
        }

    def _attention_stage_candidates(self, stage: str, *, multi: bool = False) -> list[float]:
        if multi:
            cfg = self.multi_structure_budget_controller
            floor, cap = cfg["attention_stage_bounds"][stage]
            ratios = cfg["attention_action_ratios"]
        else:
            floor, cap = self.attention_head_pruning["stage_ratio_bounds"][stage]
            ratios = self.attention_head_pruning["action_ratios"]
        return [
            float(ratio)
            for ratio in ratios
            if ratio == 0.0 or (floor - 1e-12 <= float(ratio) <= cap + 1e-12)
        ]

    def _multi_mlp_candidates(self, stage: str) -> list[float]:
        cfg = self.multi_structure_budget_controller
        floor, cap = cfg["mlp_stage_bounds"][stage]
        return [
            float(ratio)
            for ratio in cfg["mlp_action_ratios"]
            if ratio == 0.0 or (floor - 1e-12 <= float(ratio) <= cap + 1e-12)
        ]

    def _active_attention_ratio(self, stage: str) -> float:
        if self.multi_structure_budget_controller_enabled:
            return float(self._multi_selected_attention_ratios.get(stage, 0.0))
        if bool(self.attention_head_pruning.get("dynamic_budget", False)):
            return float(self._attention_selected_ratios.get(stage, 0.0))
        return float(self.attention_head_pruning["stage_ratios"].get(stage, 0.0))

    def _attention_head_score(self, layer_id: int, hidden: torch.Tensor) -> torch.Tensor:
        num_heads = self._attention_num_heads_by_layer[layer_id]
        head_dim = self._attention_head_dim_by_layer[layer_id]
        values = hidden.detach().float().reshape(*hidden.shape[:-1], num_heads, head_dim)
        activation_energy = values.square().sum(dim=-1).mean(dim=tuple(range(values.ndim - 2)))
        norms = self._attention_o_proj_norms[layer_id].to(
            device=activation_energy.device, dtype=torch.float32
        )
        return activation_energy * norms

    def _attention_append_recent(self, stage: str, layer_id: int, hidden: torch.Tensor) -> None:
        score = self._attention_head_score(layer_id, hidden)
        if int(hidden.shape[1]) > 1:
            previous = self._attention_prompt_scores.get(layer_id)
            self._attention_prompt_scores[layer_id] = (
                score if previous is None else previous.to(score.device) + score
            )
            return
        by_layer = self._attention_recent_scores[stage]
        queue = by_layer.get(layer_id)
        if queue is None:
            queue = deque()
            by_layer[layer_id] = queue
        maxlen = self.window_tokens[stage]
        current_sum = self._attention_recent_sums[stage].get(layer_id)
        if current_sum is None:
            current_sum = torch.zeros_like(score)
        else:
            current_sum = current_sum.to(score.device)
        while len(queue) >= maxlen:
            current_sum = current_sum - queue.popleft().to(score.device)
        queue.append(score.detach().cpu())
        self._attention_recent_sums[stage][layer_id] = current_sum + score

    def _attention_score_for_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        score = self._attention_recent_sums[stage].get(layer_id)
        if score is None:
            score = self._attention_prompt_scores.get(layer_id)
        if score is None:
            heads = self._attention_num_heads_by_layer[layer_id]
            score = torch.ones(heads, dtype=torch.float32)
        return score.float()

    def _record_attention_pruning(self, stage: str, layer_id: int, token_count: int, mask: torch.Tensor | None) -> None:
        ratio = 0.0 if mask is None else _mask_pruning_ratio(mask)
        count = int(token_count)
        self._attention_actual_weighted_sum += ratio * count
        self._attention_actual_denominator += count
        key = stage if stage in STAGES else "dense"
        self._attention_actual_weighted_sum_by_stage[key] += ratio * count
        self._attention_actual_denominator_by_stage[key] += count

    def _attention_achieved_pruning(self) -> float:
        if self._attention_actual_denominator <= 0:
            return 0.0
        return float(self._attention_actual_weighted_sum / self._attention_actual_denominator)

    def _estimate_attention_actual_for_ratio(self, layer_id: int, ratio: float) -> float:
        heads = int(self._attention_num_heads_by_layer.get(layer_id, 0))
        if heads <= 1:
            return 0.0
        return float(min(heads - 1, max(0, int(round(heads * float(ratio)))))) / float(heads)

    def _attention_risk(self, stage: str, layer_id: int, ratio: float) -> dict[str, float]:
        score = _zscore(self._attention_score_for_mask(stage, layer_id)).detach().cpu()
        previous = self._attention_previous_scores.get(stage)
        volatility = 0.0
        if previous is not None and previous.numel() == score.numel():
            volatility = float(
                1.0
                - torch.nn.functional.cosine_similarity(
                    score.unsqueeze(0), previous.to(score.device).unsqueeze(0)
                ).item()
            )
        self._attention_previous_scores[stage] = score
        heads = int(score.numel())
        prune = min(heads - 1, max(0, int(round(heads * float(ratio)))))
        if prune <= 0 or prune >= heads:
            margin = 1.0
        else:
            ordered = torch.sort(score, descending=False).values
            margin = float((ordered[prune] - ordered[prune - 1]).abs().item())
        margin_risk = float(1.0 / (1.0 + max(0.0, margin)))
        volatility_risk = float(max(0.0, min(1.0, volatility)))
        return {
            "attention_margin_risk": margin_risk,
            "attention_volatility_risk": volatility_risk,
            "attention_base_risk": min(1.0, 0.55 * margin_risk + 0.35 * volatility_risk),
        }

    def _choose_attention_budget_ratio(self, stage: str, layer_id: int) -> dict[str, Any]:
        candidates = self._attention_stage_candidates(stage)
        if not candidates:
            return {"selected_ratio": 0.0, "reason": "no_attention_candidate"}
        target = float(self.attention_head_pruning["target_attention_pruning"])
        debt = target - self._attention_achieved_pruning()
        desired = target + float(self.attention_head_pruning["budget_debt_gain"]) * debt
        probe = max(candidates)
        risks = self._attention_risk(stage, layer_id, probe)
        desired -= float(self.attention_head_pruning["risk_ratio_penalty"]) * risks["attention_base_risk"]
        selected = min(
            candidates,
            key=lambda value: (
                abs(self._estimate_attention_actual_for_ratio(layer_id, value) - desired),
                -value if debt >= 0.0 else value,
            ),
        )
        return {
            "selected_ratio": float(selected),
            "target_attention_pruning": target,
            "achieved_attention_pruning": self._attention_achieved_pruning(),
            "attention_budget_debt": debt,
            "desired_attention_pruning": desired,
            "candidate_ratios": candidates,
            "selected_estimated_attention_pruning": self._estimate_attention_actual_for_ratio(layer_id, selected),
            **risks,
        }

    def _maybe_update_attention_budget_decision(self, stage: str, layer_id: int) -> None:
        if (
            not self.attention_head_pruning_enabled
            or not bool(self.attention_head_pruning.get("dynamic_budget", False))
            or self.multi_structure_budget_controller_enabled
            or layer_id != 0
        ):
            return
        self._attention_tokens_since_decision[stage] += 1
        window = int(self.attention_head_pruning["decision_window_tokens"])
        if self._attention_tokens_since_decision[stage] < window:
            return
        decision = self._choose_attention_budget_ratio(stage, layer_id)
        selected = float(decision["selected_ratio"])
        if selected != self._attention_selected_ratios.get(stage):
            self._attention_clear_stage_cache(stage)
        self._attention_selected_ratios[stage] = selected
        self._attention_tokens_since_decision[stage] = 0
        self._attention_decision_log.append({"stage": stage, **decision})
        self._attention_budget_debts.append(float(decision.get("attention_budget_debt", 0.0)))

    def _maybe_update_multi_structure_decision(self, stage: str, layer_id: int) -> None:
        if not self.multi_structure_budget_controller_enabled or layer_id != 0:
            return
        self._multi_tokens_since_decision[stage] += 1
        window = int(self.multi_structure_budget_controller["decision_window_tokens"])
        if self._multi_tokens_since_decision[stage] < window:
            return
        cfg = self.multi_structure_budget_controller
        mlp_candidates = self._multi_mlp_candidates(stage)
        attention_candidates = self._attention_stage_candidates(stage, multi=True)
        mlp_target = float(cfg["mlp_target_actual_pruning"])
        attention_target = float(cfg["attention_target_actual_pruning"])
        mlp_debt = mlp_target - self._budget_achieved_pruning()
        attention_debt = attention_target - self._attention_achieved_pruning()
        mlp_desired = mlp_target + float(cfg["budget_debt_gain"]) * mlp_debt
        attention_desired = (
            attention_target
            + float(cfg["budget_debt_gain"]) * attention_debt
            - float(cfg["stage_attention_penalty"][stage])
        )
        mlp_probe = max([value for value in mlp_candidates if value > 0.0] or [0.0])
        mlp_risks = self._budget_score_components(stage, layer_id, mlp_probe)
        attention_probe = max(attention_candidates or [0.0])
        attention_risks = self._attention_risk(stage, layer_id, attention_probe)
        mlp_desired -= float(cfg["risk_ratio_penalty"]) * mlp_risks["base_risk"]
        attention_desired -= float(cfg["risk_ratio_penalty"]) * attention_risks["attention_base_risk"]
        selected_mlp = min(
            mlp_candidates,
            key=lambda value: (
                abs(self._estimate_actual_pruning_for_ratio(stage, value) - mlp_desired),
                -value if mlp_debt >= 0.0 else value,
            ),
        )
        selected_attention = min(
            attention_candidates,
            key=lambda value: (
                abs(self._estimate_attention_actual_for_ratio(layer_id, value) - attention_desired),
                -value if attention_debt >= 0.0 else value,
            ),
        )
        if selected_mlp != self._multi_selected_mlp_ratios.get(stage):
            self._clear_stage_cache(stage)
        if selected_attention != self._multi_selected_attention_ratios.get(stage):
            self._attention_clear_stage_cache(stage)
        self._multi_selected_mlp_ratios[stage] = float(selected_mlp)
        self._multi_selected_attention_ratios[stage] = float(selected_attention)
        self._multi_tokens_since_decision[stage] = 0
        self._multi_decision_log.append(
            {
                "stage": stage,
                "selected_mlp_ratio": float(selected_mlp),
                "selected_attention_ratio": float(selected_attention),
                "mlp_target_actual_pruning": mlp_target,
                "attention_target_actual_pruning": attention_target,
                "mlp_achieved_pruning": self._budget_achieved_pruning(),
                "attention_achieved_pruning": self._attention_achieved_pruning(),
                "mlp_budget_debt": mlp_debt,
                "attention_budget_debt": attention_debt,
                "mlp_desired_actual_pruning": mlp_desired,
                "attention_desired_actual_pruning": attention_desired,
                "mlp_candidates": mlp_candidates,
                "attention_candidates": attention_candidates,
                **mlp_risks,
                **attention_risks,
            }
        )

    def _single_token_mode(self, layer_id: int, token_count: int) -> tuple[str | None, bool]:
        if layer_id == 0:
            stage = self.active_stage
            observe = False
            if stage in STAGES and self._active_stage_ratio(stage) > 0.0:
                observe = self._stage_observed_tokens[stage] < self.warmup_tokens[stage]
                if observe:
                    self._stage_observed_tokens[stage] += int(token_count)
                else:
                    interval = self.refresh_intervals[stage]
                    if interval > 0 and self._stage_tokens_since_refresh[stage] >= interval:
                        self._clear_stage_cache(stage)
                        self._stage_tokens_since_refresh[stage] = 0
            self._current_single_stage = stage
            self._current_single_observe = observe
        return self._current_single_stage, self._current_single_observe

    def _record_actual_pruning(
        self,
        stage: str,
        layer_id: int,
        token_count: int,
        mask: torch.Tensor | None,
    ) -> None:
        ratio = 0.0 if mask is None else _mask_pruning_ratio(mask)
        count = int(token_count)
        self._actual_pruning_weighted_sum += ratio * count
        self._actual_pruning_denominator += count
        key = stage if stage in STAGES else "dense"
        self._actual_pruning_weighted_sum_by_stage[key] += ratio * count
        self._actual_pruning_denominator_by_stage[key] += count

    def _budget_achieved_pruning(self) -> float:
        if self._actual_pruning_denominator <= 0:
            return 0.0
        return float(self._actual_pruning_weighted_sum / self._actual_pruning_denominator)

    def _budget_score_components(self, stage: str, layer_id: int, candidate_ratio: float) -> dict[str, float]:
        entry = self.bank["policies"][self.prior_policy][stage][layer_id]
        prior = entry["metric"].float()
        runtime_score = self._recent_sums[stage].get(layer_id)
        if runtime_score is None:
            runtime_score = self._prompt_scores.get(layer_id)
        if runtime_score is None:
            runtime_score = torch.ones_like(prior)
        runtime_score = runtime_score.to(device=prior.device, dtype=torch.float32)
        current_score = runtime_score.sqrt() if self.score_mode == "activation" else runtime_score
        normalized = _zscore(current_score).detach().cpu()
        previous = self._budget_previous_scores.get(stage)
        if previous is None or previous.numel() != normalized.numel():
            volatility = 0.0
        else:
            volatility = float(
                1.0
                - torch.nn.functional.cosine_similarity(
                    normalized.unsqueeze(0),
                    previous.to(normalized.device).unsqueeze(0),
                ).item()
            )
        self._budget_previous_scores[stage] = normalized
        protected = _topk_mask(prior, self.protected_core_ratios[stage])
        prune_score = (
            self.prior_weight * _zscore(-prior)
            + self.runtime_weight * _zscore(-current_score.to(device=prior.device))
        )
        candidates = prune_score[~protected.to(device=prune_score.device, dtype=torch.bool)]
        prune_count = min(
            max(0, int(round(prune_score.numel() * float(candidate_ratio)))),
            int(candidates.numel()),
        )
        if prune_count <= 0 or candidates.numel() <= 1 or prune_count >= candidates.numel():
            margin = 1.0
        else:
            ordered = torch.sort(candidates, descending=True).values
            margin = float((ordered[prune_count - 1] - ordered[prune_count]).abs().item())
        margin_risk = float(1.0 / (1.0 + max(0.0, margin)))
        volatility_risk = float(max(0.0, min(1.0, volatility)))
        base_risk = float(
            max(
                0.0,
                min(
                    1.0,
                    0.45 * margin_risk
                    + 0.35 * volatility_risk
                    + float(self.stage_budget_controller["stage_risk_bias"][stage]),
                ),
            )
        )
        return {
            "base_risk": base_risk,
            "margin_risk": margin_risk,
            "volatility_risk": volatility_risk,
        }

    def _estimate_actual_pruning_for_ratio(self, stage: str, ratio: float) -> float:
        """Estimate realized mask sparsity after protected-core constraints."""

        layer_values: list[float] = []
        for layer_id in self.bank["layers"]:
            entry = self.bank["policies"][self.prior_policy][stage][layer_id]
            prior = entry["metric"].float()
            protected = _topk_mask(prior, self.protected_core_ratios[stage])
            channels = int(prior.numel())
            prune_count = min(
                channels - 1,
                max(0, int(round(channels * float(ratio)))),
                int((~protected).sum().item()),
            )
            layer_values.append(float(prune_count) / float(channels) if channels else 0.0)
        return sum(layer_values) / len(layer_values) if layer_values else 0.0

    def _budget_candidate_actual_estimates(
        self,
        stage: str,
        candidates: list[float],
    ) -> dict[float, float]:
        return {
            float(ratio): self._estimate_actual_pruning_for_ratio(stage, float(ratio))
            for ratio in candidates
        }

    def _choose_budget_ratio(self, stage: str, layer_id: int) -> dict[str, Any]:
        candidates = self._stage_budget_candidates(stage)
        if not candidates:
            selected = 0.0
            return {"selected_ratio": selected, "reason": "no_valid_budget_candidate"}
        achieved = self._budget_achieved_pruning()
        target = float(self.stage_budget_controller["target_actual_pruning"])
        debt = target - achieved
        nonzero_candidates = [ratio for ratio in candidates if ratio > 0.0]
        probe_ratio = max(nonzero_candidates) if nonzero_candidates else 0.0
        risks = self._budget_score_components(stage, layer_id, probe_ratio)
        desired = target + float(self.stage_budget_controller["budget_debt_gain"]) * debt
        if risks["base_risk"] <= float(self.stage_budget_controller["risk_safe_threshold"]):
            desired += 0.05
        elif risks["base_risk"] >= float(self.stage_budget_controller["risk_reduce_threshold"]):
            desired -= float(self.stage_budget_controller["risk_ratio_penalty"])
        if debt > float(self.stage_budget_controller["budget_catchup_threshold"]) and nonzero_candidates:
            desired = max(desired, min(nonzero_candidates, key=lambda value: abs(value - target)))
        ratio_selection_mode = str(self.stage_budget_controller["ratio_selection_mode"])
        if ratio_selection_mode not in {"nominal", "estimated_actual"}:
            raise ValueError(f"Unknown stage budget ratio_selection_mode: {ratio_selection_mode}")
        actual_estimates: dict[float, float] = {}
        if ratio_selection_mode == "estimated_actual":
            actual_estimates = self._budget_candidate_actual_estimates(stage, candidates)
            selected = min(
                candidates,
                key=lambda value: (
                    abs(actual_estimates.get(float(value), 0.0) - desired),
                    -value if debt >= 0.0 else value,
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda value: (abs(value - desired), -value if debt >= 0.0 else value),
            )
        if (
            risks["base_risk"] >= float(self.stage_budget_controller["risk_dense_threshold"])
            and debt <= 0.0
            and 0.0 in candidates
        ):
            selected = 0.0
        if debt > float(self.stage_budget_controller["budget_catchup_threshold"]) and nonzero_candidates:
            if ratio_selection_mode == "estimated_actual":
                selected = max(
                    selected,
                    min(
                        nonzero_candidates,
                        key=lambda value: abs(actual_estimates.get(float(value), 0.0) - target),
                    ),
                )
            else:
                selected = max(selected, min(nonzero_candidates, key=lambda value: abs(value - target)))
        return {
            "selected_ratio": float(selected),
            "target_actual_pruning": target,
            "achieved_actual_pruning": achieved,
            "budget_debt": debt,
            "desired_ratio": desired,
            "desired_actual_pruning": (
                desired if ratio_selection_mode == "estimated_actual" else None
            ),
            "candidate_ratios": candidates,
            "ratio_selection_mode": ratio_selection_mode,
            "candidate_actual_pruning_estimates": {
                f"{ratio:.2f}": actual
                for ratio, actual in sorted(actual_estimates.items())
            },
            "selected_estimated_actual_pruning": (
                actual_estimates.get(float(selected))
                if ratio_selection_mode == "estimated_actual"
                else None
            ),
            "actual_calibrated_debt": debt,
            **risks,
        }

    def _maybe_update_budget_decision(self, stage: str, layer_id: int) -> None:
        if not self.stage_budget_controller_enabled or layer_id != 0:
            return
        self._budget_tokens_since_decision[stage] += 1
        window = int(self.stage_budget_controller["decision_window_tokens"])
        if self._budget_tokens_since_decision[stage] < window:
            return
        decision = self._choose_budget_ratio(stage, layer_id)
        selected = float(decision["selected_ratio"])
        if selected != self._budget_selected_ratios.get(stage):
            self._clear_stage_cache(stage)
        self._budget_selected_ratios[stage] = selected
        self._budget_tokens_since_decision[stage] = 0
        self._budget_decision_log.append({"stage": stage, **decision})
        self._budget_base_risks[stage].append(float(decision.get("base_risk", 0.0)))
        self._budget_margin_risks[stage].append(float(decision.get("margin_risk", 0.0)))
        self._budget_volatility_risks[stage].append(float(decision.get("volatility_risk", 0.0)))
        self._budget_debts.append(float(decision.get("budget_debt", 0.0)))

    def observe_or_mask(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor | None:
        token_count = int(intermediate.shape[1])
        if token_count > 1:
            self.observe_prompt(layer_id, intermediate)
            return None
        stage, observe = self._single_token_mode(layer_id, token_count)
        if self.fallback_reason is not None or stage not in STAGES:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage["dense"] += token_count
            self._record_actual_pruning("dense", layer_id, token_count, None)
            return None
        self._append_recent(stage, layer_id, intermediate)
        if layer_id == 0:
            if self.multi_structure_budget_controller_enabled:
                self._maybe_update_multi_structure_decision(stage, layer_id)
            else:
                self._maybe_update_budget_decision(stage, layer_id)
        ratio = self._active_stage_ratio(stage)
        if observe:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            self._record_actual_pruning(stage, layer_id, token_count, None)
            return None
        if ratio <= 0.0:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
                if self.stage_budget_controller_enabled:
                    self._budget_ratio_tokens[f"{stage}:0.00"] += token_count
            self._record_actual_pruning(stage, layer_id, token_count, None)
            return None
        if layer_id == 0:
            self.masked_tokens_by_stage[stage] += token_count
            self._stage_tokens_since_refresh[stage] += token_count
            if self.stage_budget_controller_enabled:
                self._budget_ratio_tokens[f"{stage}:{ratio:.2f}"] += token_count
        mask = self.keep_mask(stage, layer_id)
        self._record_actual_pruning(stage, layer_id, token_count, mask)
        return mask

    def observe_or_mask_attention(self, layer_id: int, hidden: torch.Tensor) -> torch.Tensor | None:
        if not self.attention_head_pruning_enabled:
            return None
        token_count = int(hidden.shape[1])
        if layer_id not in self._attention_num_heads_by_layer:
            raise RuntimeError(f"Attention layer {layer_id} was not registered")
        if token_count > 1:
            self._attention_append_recent("prompt", layer_id, hidden)
            return None
        stage = self.active_stage
        if self.fallback_reason is not None or stage not in STAGES:
            self._record_attention_pruning("dense", layer_id, token_count, None)
            return None
        self._attention_append_recent(stage, layer_id, hidden)
        if layer_id == 0:
            self._maybe_update_attention_budget_decision(stage, layer_id)
        ratio = self._active_attention_ratio(stage)
        if ratio <= 0.0:
            if layer_id == 0:
                self._attention_ratio_tokens[f"{stage}:0.00"] += token_count
            self._record_attention_pruning(stage, layer_id, token_count, None)
            return None
        if layer_id == 0:
            self._attention_ratio_tokens[f"{stage}:{ratio:.4f}"] += token_count
        mask = self.keep_attention_head_mask(stage, layer_id)
        self._record_attention_pruning(stage, layer_id, token_count, mask)
        return mask

    def keep_attention_head_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        key = (stage, layer_id)
        cached = self._attention_mask_cache.get(key)
        if cached is not None:
            return cached
        score = self._attention_score_for_mask(stage, layer_id)
        mask = _head_mask_from_scores(score, self._active_attention_ratio(stage))
        self._attention_mask_cache[key] = mask
        return mask

    def keep_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        key = (stage, layer_id)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        entry = self.bank["policies"][self.prior_policy][stage][layer_id]
        prior = entry["metric"].float()
        runtime_score = self._recent_sums[stage].get(layer_id)
        if runtime_score is None:
            runtime_score = self._prompt_scores.get(layer_id)
        if runtime_score is None:
            runtime_score = torch.ones_like(prior)
        runtime_score = runtime_score.to(device=prior.device, dtype=torch.float32)
        if self.score_mode == "activation":
            runtime_score = runtime_score.sqrt()
        protected = _topk_mask(prior, self.protected_core_ratios[stage])
        prune_score = self.prior_weight * _zscore(-prior) + self.runtime_weight * _zscore(-runtime_score)
        mask = _prune_mask_from_scores(
            prune_score,
            self._active_stage_ratio(stage),
            protected_mask=protected,
        )
        mask = self._limit_mask_swaps(stage, layer_id, mask, protected, -prune_score)
        self._mask_cache[key] = mask
        if layer_id == 0:
            self.mask_refresh_count_by_stage[stage] += 1
        return mask

    def _limit_mask_swaps(
        self,
        stage: str,
        layer_id: int,
        candidate: torch.Tensor,
        protected: torch.Tensor,
        keep_score: torch.Tensor,
    ) -> torch.Tensor:
        """Limit optional replacements while preserving the candidate keep count."""

        previous = self._previous_masks.get(layer_id)
        key = f"{stage}:{layer_id}"
        if previous is None or previous.numel() != candidate.numel():
            self._previous_masks[layer_id] = candidate.detach().cpu()
            return candidate

        previous = previous.to(device=candidate.device, dtype=torch.bool)
        target_keep = int(candidate.sum().item())
        current = previous.clone()
        protected = protected.to(device=current.device, dtype=torch.bool)
        union = torch.logical_or(previous, candidate).sum().item()
        self._mask_jaccard_sum_by_stage_layer[key] += (
            float(torch.logical_and(previous, candidate).sum().item() / union) if union else 1.0
        )
        self._mask_jaccard_count_by_stage_layer[key] += 1

        # Protection and target cardinality take precedence over continuity.
        current |= protected
        if int(current.sum().item()) > target_keep:
            removable = current & ~protected
            remove_count = int(current.sum().item()) - target_keep
            drop = _topk_within_mask(-keep_score, removable, remove_count)
            current[drop] = False
        elif int(current.sum().item()) < target_keep:
            add_count = target_keep - int(current.sum().item())
            add = _topk_within_mask(keep_score, ~current, add_count)
            current[add] = True

        drop_candidates = current & ~candidate & ~protected
        add_candidates = ~current & candidate
        candidate_pairs = min(
            int(drop_candidates.sum().item()), int(add_candidates.sum().item())
        )
        self._mask_swap_candidates_by_stage_layer[key] += candidate_pairs
        allowed_pairs = int(candidate.numel() * self.max_mask_swap_fraction)
        swap_count = min(candidate_pairs, allowed_pairs)
        if swap_count:
            drop = _topk_within_mask(-keep_score, drop_candidates, swap_count)
            add = _topk_within_mask(keep_score, add_candidates, swap_count)
            current[drop] = False
            current[add] = True
        self._mask_swap_pairs_by_stage_layer[key] += swap_count
        self._previous_masks[layer_id] = current.detach().cpu()
        return current

    def entry(self, stage: str, layer_id: int) -> dict[str, Any]:
        return self.bank["policies"][self.prior_policy][stage][layer_id]

    def summary(self) -> dict[str, Any]:
        total = sum(self.dense_observation_tokens_by_stage.values()) + sum(
            self.masked_tokens_by_stage.values()
        )
        weighted = sum(
            self.stage_ratios.get(stage, 0.0) * count
            for stage, count in self.masked_tokens_by_stage.items()
        )
        actual_by_stage = {
            stage: (
                float(self._actual_pruning_weighted_sum_by_stage[stage])
                / float(self._actual_pruning_denominator_by_stage[stage])
            )
            for stage in self._actual_pruning_denominator_by_stage
            if self._actual_pruning_denominator_by_stage[stage]
        }
        budget_ratio_denominator = sum(self._budget_ratio_tokens.values())
        budget_weighted = 0.0
        for key, count in self._budget_ratio_tokens.items():
            try:
                ratio = float(str(key).rsplit(":", 1)[1])
            except (IndexError, ValueError):
                ratio = 0.0
            budget_weighted += ratio * int(count)
        budget_actions_by_stage = Counter(
            str(decision.get("stage", "unknown")) for decision in self._budget_decision_log
        )
        budget_selected_by_stage = Counter(
            f"{decision.get('stage', 'unknown')}:{float(decision.get('selected_ratio', 0.0)):.2f}"
            for decision in self._budget_decision_log
        )
        budget_debt_summary = None
        if self._budget_debts:
            budget_debt_summary = {
                "min": min(self._budget_debts),
                "max": max(self._budget_debts),
                "mean": sum(self._budget_debts) / len(self._budget_debts),
                "final": self._budget_debts[-1],
            }
        attention_actual_by_stage = {
            stage: (
                float(self._attention_actual_weighted_sum_by_stage[stage])
                / float(self._attention_actual_denominator_by_stage[stage])
            )
            for stage in self._attention_actual_denominator_by_stage
            if self._attention_actual_denominator_by_stage[stage]
        }
        attention_actions_by_stage = Counter(
            str(decision.get("stage", "unknown")) for decision in self._attention_decision_log
        )
        attention_selected_by_stage = Counter(
            f"{decision.get('stage', 'unknown')}:{float(decision.get('selected_ratio', 0.0)):.4f}"
            for decision in self._attention_decision_log
        )
        multi_actions_by_stage = Counter(
            str(decision.get("stage", "unknown")) for decision in self._multi_decision_log
        )
        multi_selected_pairs = Counter(
            (
                f"{decision.get('stage', 'unknown')}:"
                f"mlp={float(decision.get('selected_mlp_ratio', 0.0)):.2f},"
                f"attn={float(decision.get('selected_attention_ratio', 0.0)):.4f}"
            )
            for decision in self._multi_decision_log
        )

        def mean_values(values_by_stage: dict[str, list[float]]) -> dict[str, float]:
            return {
                stage: sum(values) / len(values)
                for stage, values in values_by_stage.items()
                if values
            }

        summary = {
            "backend": "calibrated_stage_safe_dynamic_griffin_logical_v2",
            "real_speedup_claimed": False,
            "policy": "calibrated_stage_safe_dynamic_griffin",
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "runtime_weight": self.runtime_weight,
            "prior_weight": self.prior_weight,
            "warmup_tokens": self.warmup_tokens,
            "stage_ratios": self.stage_ratios,
            "protected_core_ratios": self.protected_core_ratios,
            "refresh_intervals": self.refresh_intervals,
            "window_tokens": self.window_tokens,
            "tokens_by_stage": dict(self.tokens_by_stage),
            "dense_observation_tokens_by_stage": dict(self.dense_observation_tokens_by_stage),
            "masked_tokens_by_stage": dict(self.masked_tokens_by_stage),
            "mask_refresh_count_by_stage": dict(self.mask_refresh_count_by_stage),
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "fallback_behavior": self.fallback_behavior,
            "max_mask_swap_fraction": self.max_mask_swap_fraction,
            "output_norm_source": (
                "frozen_model_down_proj" if self._output_norm_hash is not None else None
            ),
            "output_norm_hash": self._output_norm_hash,
            "output_norm_layers": sorted(self._output_norms),
            "mask_swap_pairs_by_stage_layer": dict(self._mask_swap_pairs_by_stage_layer),
            "mask_swap_candidates_by_stage_layer": dict(
                self._mask_swap_candidates_by_stage_layer
            ),
            "mean_mask_jaccard_by_stage_layer": {
                key: self._mask_jaccard_sum_by_stage_layer[key]
                / max(1, self._mask_jaccard_count_by_stage_layer[key])
                for key in self._mask_jaccard_count_by_stage_layer
            },
            "theoretical_average_mlp_pruning_ratio": weighted / total if total else 0.0,
            "actual_average_mlp_pruning_ratio": (
                self._actual_pruning_weighted_sum / self._actual_pruning_denominator
                if self._actual_pruning_denominator
                else 0.0
            ),
            "actual_pruning_ratio_by_stage": actual_by_stage,
            "actual_pruning_accounting": "actual_mask_sparsity_layer_token_weighted",
            "attention_head_pruning_enabled": self.attention_head_pruning_enabled,
            "attention_head_score_mode": (
                self.attention_head_pruning["score_mode"]
                if self.attention_head_pruning_enabled
                else None
            ),
            "attention_o_proj_norm_hash": self._attention_o_proj_norm_hash,
            "actual_average_attention_head_pruning_ratio": (
                self._attention_actual_weighted_sum / self._attention_actual_denominator
                if self._attention_actual_denominator
                else 0.0
            ),
            "actual_attention_head_pruning_ratio_by_stage": attention_actual_by_stage,
            "attention_head_ratio_tokens": dict(self._attention_ratio_tokens),
            "attention_head_selected_ratio_actions": dict(attention_selected_by_stage),
            "attention_head_actions_by_stage": dict(attention_actions_by_stage),
            "attention_head_budget_decisions": list(self._attention_decision_log),
            "multi_structure_budget_controller_enabled": self.multi_structure_budget_controller_enabled,
            "multi_structure_budget_decisions": list(self._multi_decision_log),
            "multi_structure_actions_by_stage": dict(multi_actions_by_stage),
            "multi_structure_selected_pair_actions": dict(multi_selected_pairs),
            "selected_mlp_ratio_by_stage": dict(self._multi_selected_mlp_ratios),
            "selected_attention_ratio_by_stage": dict(self._multi_selected_attention_ratios),
            "joint_logical_pruning_summary": {
                "mlp": (
                    self._actual_pruning_weighted_sum / self._actual_pruning_denominator
                    if self._actual_pruning_denominator
                    else 0.0
                ),
                "attention_head": (
                    self._attention_actual_weighted_sum / self._attention_actual_denominator
                    if self._attention_actual_denominator
                    else 0.0
                ),
            },
        }
        if self.stage_budget_controller_enabled:
            summary.update(
                {
                    "stage_budget_controller_enabled": True,
                    "stage_budget_controller": {
                        "target_actual_pruning": self.stage_budget_controller[
                            "target_actual_pruning"
                        ],
                        "decision_window_tokens": self.stage_budget_controller[
                            "decision_window_tokens"
                        ],
                        "action_ratios": list(self.stage_budget_controller["action_ratios"]),
                        "stage_ratio_bounds": {
                            stage: list(bounds)
                            for stage, bounds in self.stage_budget_controller[
                                "stage_ratio_bounds"
                            ].items()
                        },
                        "stage_risk_bias": dict(self.stage_budget_controller["stage_risk_bias"]),
                        "risk_dense_threshold": self.stage_budget_controller[
                            "risk_dense_threshold"
                        ],
                        "risk_reduce_threshold": self.stage_budget_controller[
                            "risk_reduce_threshold"
                        ],
                        "risk_safe_threshold": self.stage_budget_controller[
                            "risk_safe_threshold"
                        ],
                        "budget_debt_gain": self.stage_budget_controller["budget_debt_gain"],
                        "budget_catchup_threshold": self.stage_budget_controller[
                            "budget_catchup_threshold"
                        ],
                        "risk_ratio_penalty": self.stage_budget_controller[
                            "risk_ratio_penalty"
                        ],
                        "ratio_selection_mode": self.stage_budget_controller[
                            "ratio_selection_mode"
                        ],
                    },
                    "stage_budget_selected_ratios": dict(self._budget_selected_ratios),
                    "stage_budget_decisions": list(self._budget_decision_log),
                    "stage_budget_actions_by_stage": dict(budget_actions_by_stage),
                    "stage_budget_selected_ratio_actions": dict(budget_selected_by_stage),
                    "stage_budget_ratio_tokens": dict(self._budget_ratio_tokens),
                    "stage_budget_theoretical_selected_ratio": (
                        budget_weighted / budget_ratio_denominator
                        if budget_ratio_denominator
                        else 0.0
                    ),
                    "stage_budget_mean_base_risk_by_stage": mean_values(
                        self._budget_base_risks
                    ),
                    "stage_budget_mean_margin_risk_by_stage": mean_values(
                        self._budget_margin_risks
                    ),
                    "stage_budget_mean_volatility_risk_by_stage": mean_values(
                        self._budget_volatility_risks
                    ),
                    "stage_budget_debt_summary": budget_debt_summary,
                }
            )
        else:
            summary["stage_budget_controller_enabled"] = False
        if self.attention_head_pruning_enabled:
            summary["attention_head_pruning"] = {
                "enabled": True,
                "score_mode": self.attention_head_pruning["score_mode"],
                "stage_ratios": dict(self.attention_head_pruning["stage_ratios"]),
                "action_ratios": list(self.attention_head_pruning["action_ratios"]),
                "stage_ratio_bounds": {
                    stage: list(bounds)
                    for stage, bounds in self.attention_head_pruning["stage_ratio_bounds"].items()
                },
                "target_attention_pruning": self.attention_head_pruning[
                    "target_attention_pruning"
                ],
                "decision_window_tokens": self.attention_head_pruning[
                    "decision_window_tokens"
                ],
                "dynamic_budget": self.attention_head_pruning["dynamic_budget"],
            }
        if self.multi_structure_budget_controller_enabled:
            summary["multi_structure_budget_controller"] = {
                "enabled": True,
                "decision_window_tokens": self.multi_structure_budget_controller[
                    "decision_window_tokens"
                ],
                "mlp_target_actual_pruning": self.multi_structure_budget_controller[
                    "mlp_target_actual_pruning"
                ],
                "attention_target_actual_pruning": self.multi_structure_budget_controller[
                    "attention_target_actual_pruning"
                ],
                "mlp_action_ratios": list(
                    self.multi_structure_budget_controller["mlp_action_ratios"]
                ),
                "attention_action_ratios": list(
                    self.multi_structure_budget_controller["attention_action_ratios"]
                ),
                "mlp_stage_bounds": {
                    stage: list(bounds)
                    for stage, bounds in self.multi_structure_budget_controller[
                        "mlp_stage_bounds"
                    ].items()
                },
                "attention_stage_bounds": {
                    stage: list(bounds)
                    for stage, bounds in self.multi_structure_budget_controller[
                        "attention_stage_bounds"
                    ].items()
                },
                "stage_attention_penalty": dict(
                    self.multi_structure_budget_controller["stage_attention_penalty"]
                ),
            }
        return summary


class StageRiskAdaptiveRuntime(SafeDynamicStageGriffinRuntime):
    """Stage-conditioned budget controller over the safe dynamic channel mask.

    Stage remains an explicit, causal conditioning variable.  The learned
    controller only selects a safe budget for a stage-local decode window; it
    never predicts or replaces the stage protocol itself.
    """

    def __init__(
        self,
        bank: dict[str, Any],
        *,
        stage_ratios: dict[str, float],
        controller_checkpoint_path: str,
        action_ratios: list[float],
        risk_thresholds: dict[str, float],
        stage_ratio_caps: dict[str, float],
        min_warmup_tokens: dict[str, int] | None = None,
        decision_window_tokens: int = 64,
        target_actual_pruning: float = 0.34,
        max_mask_swap_fraction: float = 0.05,
        score_mode: str = "current_safe",
        fallback_behavior: str = "dense_after_error",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bank,
            stage_ratios=stage_ratios,
            fallback_behavior=fallback_behavior,
            **kwargs,
        )
        self.action_ratios = tuple(sorted({float(value) for value in action_ratios}))
        if not self.action_ratios or self.action_ratios[0] != 0.0:
            raise ValueError("Stage-risk action ratios must start with dense ratio 0.0")
        if any(not 0.0 <= value < 1.0 for value in self.action_ratios):
            raise ValueError("Invalid stage-risk action ratio")
        if decision_window_tokens < 1:
            raise ValueError("Stage-risk decision window must be positive")
        if not 0.0 <= target_actual_pruning < 1.0:
            raise ValueError("Invalid stage-risk pruning target")
        if not 0.0 <= max_mask_swap_fraction <= 1.0:
            raise ValueError("Invalid mask swap fraction")
        if score_mode not in {"current_safe", "output_aware"}:
            raise ValueError("Unsupported stage-risk score mode")
        if fallback_behavior not in {"dense_after_error", "keep_masking_after_error"}:
            raise ValueError("Unsupported stage-risk fallback behavior")
        self.controller = StageRiskController(
            controller_checkpoint_path,
            tuple(self.action_ratios),
            {stage: float(risk_thresholds[stage]) for stage in STAGES},
        )
        if score_mode == "output_aware":
            missing = [
                (stage, layer_id)
                for stage in STAGES
                for layer_id in self.bank["layers"]
                if "output_norm" not in self.bank["policies"][self.prior_policy][stage][layer_id]
            ]
            if missing:
                raise ValueError(
                    "output_aware requires a v1 stage-risk bank with explicit down_proj output norms"
                )
        self.stage_ratio_caps = {stage: float(stage_ratio_caps[stage]) for stage in STAGES}
        if any(value < 0.0 or value >= 1.0 for value in self.stage_ratio_caps.values()):
            raise ValueError("Invalid stage-risk stage cap")
        self.decision_window_tokens = int(decision_window_tokens)
        min_warmup_tokens = min_warmup_tokens or kwargs.get("warmup_tokens") or {}
        self.min_warmup_tokens = {
            stage: max(0, int(min_warmup_tokens.get(stage, 0))) for stage in STAGES
        }
        self.target_actual_pruning = float(target_actual_pruning)
        self.max_mask_swap_fraction = float(max_mask_swap_fraction)
        self.score_mode = _safe_dynamic_score_mode(score_mode)
        self.fallback_behavior = fallback_behavior
        self._decode_entropy = 0.0
        self._decode_confidence = 1.0
        self._stage_local_tokens: Counter[str] = Counter()
        self._tokens_since_decision: Counter[str] = Counter()
        self._selected_ratios: dict[str, float] = {stage: 0.0 for stage in STAGES}
        self._previous_masks: dict[tuple[str, int], torch.Tensor] = {}
        self._previous_runtime_scores: dict[str, torch.Tensor] = {}
        self._decision_log: list[dict[str, Any]] = []
        self._selected_ratio_tokens: Counter[str] = Counter()
        self._mask_swap_count = 0
        self._mask_swap_candidates = 0
        self._mask_jaccards: list[float] = []
        self._budget_dual = 0.0

    def reset(self) -> None:
        super().reset()
        self._decode_entropy = 0.0
        self._decode_confidence = 1.0
        self._stage_local_tokens.clear()
        self._tokens_since_decision.clear()
        self._selected_ratios = {stage: 0.0 for stage in STAGES}
        self._previous_masks.clear()
        self._previous_runtime_scores.clear()
        self._decision_log.clear()
        self._selected_ratio_tokens.clear()
        self._mask_swap_count = 0
        self._mask_swap_candidates = 0
        self._mask_jaccards.clear()
        self._budget_dual = 0.0

    def set_decode_observation(self, *, entropy: float, confidence: float) -> None:
        if torch.isfinite(torch.tensor([entropy, confidence])).all():
            self._decode_entropy = float(entropy)
            self._decode_confidence = float(confidence)

    def set_stage(self, stage: str) -> None:
        previous = self.active_stage
        super().set_stage(stage)
        if self.active_stage != previous and self.active_stage in STAGES:
            self._tokens_since_decision[self.active_stage] = self.decision_window_tokens

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_behavior == "keep_masking_after_error" and self.active_stage in STAGES:
            if self.fallback_reason is None:
                self.fallback_reason = str(reason)
            return
        super().fallback_dense(reason)

    def _runtime_features(self, stage: str, layer_id: int, intermediate: torch.Tensor, ratio: float) -> np.ndarray:
        entry = self.bank["policies"][self.prior_policy][stage][layer_id]
        prior = entry["metric"].to(device=intermediate.device, dtype=torch.float32)
        activation_energy = intermediate.detach().float().square().mean(dim=(0, 1))
        raw = activation_energy
        if self.score_mode == "output_aware":
            raw = raw * entry["output_norm"].to(device=raw.device, dtype=torch.float32)
        normalized_raw = _zscore(raw)
        activation_ood = float((_zscore(raw) - _zscore(prior)).abs().mean().item())
        previous = self._previous_runtime_scores.get(stage)
        activation_drift = (
            0.0
            if previous is None
            else float(1.0 - torch.nn.functional.cosine_similarity(
                normalized_raw.unsqueeze(0), previous.to(normalized_raw.device).unsqueeze(0)
            ).item())
        )
        self._previous_runtime_scores[stage] = normalized_raw.detach().cpu()
        protected = _topk_mask(prior, self.protected_core_ratios[stage])
        prune_score = self.prior_weight * _zscore(-prior) + self.runtime_weight * _zscore(-raw)
        candidates = prune_score[~protected]
        count = min(max(1, int(round(prune_score.numel() * ratio))), int(candidates.numel()))
        ordered = torch.sort(candidates).values
        margin = 0.0 if count >= ordered.numel() else float((ordered[count] - ordered[count - 1]).abs().item())
        return stage_risk_features(
            stage_local_tokens=int(self._stage_local_tokens[stage]),
            entropy=self._decode_entropy,
            confidence=self._decode_confidence,
            activation_ood=activation_ood,
            activation_drift=activation_drift,
            mask_score_margin=margin,
            candidate_ratio=ratio,
        )

    def _minimum_budget_ratio(self) -> float:
        observed = max(1, self._actual_pruning_denominator)
        achieved = self._actual_pruning_weighted_sum / observed
        gap = self.target_actual_pruning - achieved
        # A bounded dual variable pushes safe later windows to repay pruning
        # budget borrowed by dense high-risk windows. It never overrides risk.
        self._budget_dual = float(np.clip(self._budget_dual + 0.10 * gap, 0.0, 1.0))
        return min(self.action_ratios[-1], max(0.0, gap * 1.5 + 0.20 * self._budget_dual))

    def _select_ratio(self, stage: str, intermediate: torch.Tensor) -> None:
        cap = self.stage_ratio_caps[stage]
        candidates = [
            (ratio, self._runtime_features(stage, 0, intermediate, ratio))
            for ratio in self.action_ratios
            if 0.0 < ratio <= cap + 1e-12
        ]
        selected, detail = self.controller.choose_ratio(
            stage=stage,
            candidates=candidates,
            minimum_ratio=self._minimum_budget_ratio(),
        )
        self._selected_ratios[stage] = selected
        self._clear_stage_cache(stage)
        self._tokens_since_decision[stage] = 0
        self._decision_log.append(
            {
                "stage": stage,
                "stage_local_tokens": int(self._stage_local_tokens[stage]),
                "selected_ratio": selected,
                "score_mode": self.score_mode,
                **detail,
            }
        )

    def _limit_mask_swaps(self, stage: str, layer_id: int, mask: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        key = (stage, layer_id)
        previous = self._previous_masks.get(key)
        if previous is None or previous.numel() != mask.numel():
            self._previous_masks[key] = mask.detach().cpu()
            return mask
        previous = previous.to(mask.device)
        union = torch.logical_or(previous, mask).sum().item()
        self._mask_jaccards.append(
            float(torch.logical_and(previous, mask).sum().item() / union) if union else 1.0
        )
        changed = torch.nonzero(previous != mask, as_tuple=False).flatten()
        self._mask_swap_candidates += int(changed.numel())
        allowed = int(round(mask.numel() * self.max_mask_swap_fraction))
        if changed.numel() > allowed:
            keep_changes = torch.topk(score[changed].abs(), k=max(0, allowed), largest=True).indices
            accepted = changed[keep_changes]
            limited = previous.clone()
            limited[accepted] = mask[accepted]
            mask = limited
            changed = accepted
        self._mask_swap_count += int(changed.numel())
        self._previous_masks[key] = mask.detach().cpu()
        return mask

    def observe_or_mask(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor | None:
        token_count = int(intermediate.shape[1])
        if token_count > 1:
            self.observe_prompt(layer_id, intermediate)
            return None
        stage = self.active_stage
        if self.fallback_reason is not None and self.fallback_behavior == "dense_after_error":
            self._record_actual_pruning("dense", layer_id, token_count, None)
            return None
        if stage not in STAGES:
            self._record_actual_pruning("dense", layer_id, token_count, None)
            return None
        self._append_recent(stage, layer_id, intermediate)
        if layer_id == 0:
            self._stage_local_tokens[stage] += token_count
            self._tokens_since_decision[stage] += token_count
            if self._stage_local_tokens[stage] <= self.min_warmup_tokens[stage]:
                self.dense_observation_tokens_by_stage[stage] += token_count
            elif self._tokens_since_decision[stage] >= self.decision_window_tokens:
                self._select_ratio(stage, intermediate)
        ratio = self._selected_ratios[stage]
        if self._stage_local_tokens[stage] <= self.min_warmup_tokens[stage] or ratio <= 0.0:
            self._record_actual_pruning(stage, layer_id, token_count, None)
            return None
        if layer_id == 0:
            self.masked_tokens_by_stage[stage] += token_count
            self._selected_ratio_tokens[f"{stage}:{ratio:.2f}"] += token_count
        mask = self.keep_mask(stage, layer_id)
        self._record_actual_pruning(stage, layer_id, token_count, mask)
        return mask

    def keep_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        key = (stage, layer_id)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        entry = self.bank["policies"][self.prior_policy][stage][layer_id]
        prior = entry["metric"].float()
        runtime_score = self._recent_sums[stage].get(layer_id)
        if runtime_score is None:
            runtime_score = self._prompt_scores.get(layer_id, torch.ones_like(prior))
        runtime_score = runtime_score.to(prior.device, dtype=torch.float32).sqrt()
        if self.score_mode == "output_aware":
            runtime_score = runtime_score.square() * entry["output_norm"].to(
                device=prior.device, dtype=torch.float32
            )
        protected = _topk_mask(prior, self.protected_core_ratios[stage])
        prune_score = self.prior_weight * _zscore(-prior) + self.runtime_weight * _zscore(-runtime_score)
        mask = _prune_mask_from_scores(prune_score, self._selected_ratios[stage], protected_mask=protected)
        mask = self._limit_mask_swaps(stage, layer_id, mask, prune_score)
        self._mask_cache[key] = mask
        if layer_id == 0:
            self.mask_refresh_count_by_stage[stage] += 1
        return mask

    def summary(self) -> dict[str, Any]:
        output = super().summary()
        output.update(
            {
                "backend": "stage_risk_adaptive_logical_v1",
                "policy": "stage_risk_adaptive",
                "controller_checkpoint": str(self.controller.checkpoint_path),
                "action_ratios": list(self.action_ratios),
                "decision_window_tokens": self.decision_window_tokens,
                "target_actual_pruning": self.target_actual_pruning,
                "stage_ratio_caps": self.stage_ratio_caps,
                "min_warmup_tokens": self.min_warmup_tokens,
                "fallback_behavior": self.fallback_behavior,
                "score_mode": self.score_mode,
                "stage_local_tokens": dict(self._stage_local_tokens),
                "selected_ratio_tokens": dict(self._selected_ratio_tokens),
                "controller_decisions": list(self._decision_log),
                "mask_swap_count": self._mask_swap_count,
                "mask_swap_candidates": self._mask_swap_candidates,
                "mean_mask_jaccard": (
                    sum(self._mask_jaccards) / len(self._mask_jaccards)
                    if self._mask_jaccards
                    else 1.0
                ),
                "budget_dual": self._budget_dual,
            }
        )
        return output


class StaticCoreResidualStageRuntime:
    def __init__(
        self,
        bank: dict[str, Any],
        *,
        stage_ratios: dict[str, float],
        base_policy: str = "trajectory_global",
        stage_prior_policy: str = "stage_specific",
        runtime_weight: float = 0.3,
        prior_weight: float = 0.7,
        warmup_tokens: dict[str, int] | None = None,
        static_core_ratios: dict[str, float] | None = None,
        swap_ratios: dict[str, float] | None = None,
        refresh_intervals: dict[str, int] | None = None,
        window_tokens: dict[str, int] | None = None,
        bias_compensation: bool = True,
    ) -> None:
        validate_mask_bank(bank)
        if base_policy not in bank["policies"]:
            raise ValueError(f"Unknown residual base policy: {base_policy}")
        if stage_prior_policy not in bank["policies"]:
            raise ValueError(f"Unknown residual stage prior policy: {stage_prior_policy}")
        self.bank = bank
        self.base_policy = str(base_policy)
        self.stage_prior_policy = str(stage_prior_policy)
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        allowed = {float(value) for value in bank["ratios"]}
        if any(value not in allowed for value in self.stage_ratios.values()):
            raise ValueError("Residual stage ratio is not present in mask bank")
        self.runtime_weight = float(runtime_weight)
        self.prior_weight = float(prior_weight)
        if self.runtime_weight < 0.0 or self.prior_weight < 0.0:
            raise ValueError("runtime_weight and prior_weight must be non-negative")
        if self.runtime_weight + self.prior_weight <= 0.0:
            raise ValueError("At least one residual score weight must be positive")
        warmup_tokens = warmup_tokens or {}
        static_core_ratios = static_core_ratios or {}
        swap_ratios = swap_ratios or {}
        refresh_intervals = refresh_intervals or {}
        window_tokens = window_tokens or {}
        self.warmup_tokens = {stage: int(warmup_tokens.get(stage, 0)) for stage in STAGES}
        self.static_core_ratios = {
            stage: float(static_core_ratios.get(stage, 1.0)) for stage in STAGES
        }
        self.swap_ratios = {stage: float(swap_ratios.get(stage, 0.0)) for stage in STAGES}
        self.refresh_intervals = {
            stage: int(refresh_intervals.get(stage, 0)) for stage in STAGES
        }
        self.window_tokens = {
            stage: max(1, int(window_tokens.get(stage, self.refresh_intervals[stage] or 1)))
            for stage in STAGES
        }
        for stage, value in self.static_core_ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid static core ratio for {stage}: {value}")
        for stage, value in self.swap_ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid residual swap ratio for {stage}: {value}")
        self.bias_compensation = bool(bias_compensation)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()
        self.dense_observation_tokens_by_stage: Counter[str] = Counter()
        self.masked_tokens_by_stage: Counter[str] = Counter()
        self.mask_refresh_count_by_stage: Counter[str] = Counter()
        self._prompt_scores: dict[int, torch.Tensor] = {}
        self._recent_scores: dict[str, dict[int, deque[torch.Tensor]]] = {stage: {} for stage in STAGES}
        self._recent_sums: dict[str, dict[int, torch.Tensor]] = {stage: {} for stage in STAGES}
        self._stage_observed_tokens: Counter[str] = Counter()
        self._stage_tokens_since_refresh: Counter[str] = Counter()
        self._mask_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._swap_channels_by_stage_layer: dict[str, dict[int, int]] = {stage: {} for stage in STAGES}
        self._current_single_stage: str | None = None
        self._current_single_observe = False

    @property
    def alpha(self) -> float:
        total = self.runtime_weight + self.prior_weight
        return self.runtime_weight / total

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()
        self.dense_observation_tokens_by_stage.clear()
        self.masked_tokens_by_stage.clear()
        self.mask_refresh_count_by_stage.clear()
        self._prompt_scores.clear()
        self._recent_scores = {stage: {} for stage in STAGES}
        self._recent_sums = {stage: {} for stage in STAGES}
        self._stage_observed_tokens.clear()
        self._stage_tokens_since_refresh.clear()
        self._mask_cache.clear()
        self._swap_channels_by_stage_layer = {stage: {} for stage in STAGES}
        self._current_single_stage = None
        self._current_single_observe = False

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            self.fallback_dense(f"unknown_stage:{stage}")
            return
        if self.fallback_reason is None:
            self.active_stage = stage
            self._current_single_stage = None
            self._current_single_observe = False

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        self.active_stage = None
        self._current_single_stage = None
        self._current_single_observe = False

    def record_token(self) -> None:
        self.tokens_by_stage[self.active_stage or "dense"] += 1

    def _accumulate_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        score_sq = griffin_activation_score(intermediate).square()
        previous = self._prompt_scores.get(layer_id)
        self._prompt_scores[layer_id] = score_sq if previous is None else previous.to(score_sq.device) + score_sq

    def _append_recent(self, stage: str, layer_id: int, intermediate: torch.Tensor) -> None:
        score_sq = griffin_activation_score(intermediate).square()
        by_layer = self._recent_scores[stage]
        queue = by_layer.get(layer_id)
        if queue is None:
            queue = deque()
            by_layer[layer_id] = queue
        maxlen = self.window_tokens[stage]
        current_sum = self._recent_sums[stage].get(layer_id)
        if current_sum is None:
            current_sum = torch.zeros_like(score_sq)
        else:
            current_sum = current_sum.to(score_sq.device)
        while len(queue) >= maxlen:
            current_sum = current_sum - queue.popleft().to(score_sq.device)
        queue.append(score_sq.detach().cpu())
        self._recent_sums[stage][layer_id] = current_sum + score_sq

    def observe_prompt(self, layer_id: int, intermediate: torch.Tensor) -> None:
        self._accumulate_prompt(layer_id, intermediate)

    def _clear_stage_cache(self, stage: str) -> None:
        self._mask_cache = {key: value for key, value in self._mask_cache.items() if key[0] != stage}

    def _single_token_mode(self, layer_id: int, token_count: int) -> tuple[str | None, bool]:
        if layer_id == 0:
            stage = self.active_stage
            observe = False
            if stage in STAGES and self.stage_ratios[stage] > 0.0:
                observe = self._stage_observed_tokens[stage] < self.warmup_tokens[stage]
                if observe:
                    self._stage_observed_tokens[stage] += int(token_count)
                else:
                    interval = self.refresh_intervals[stage]
                    if interval > 0 and self._stage_tokens_since_refresh[stage] >= interval:
                        self._clear_stage_cache(stage)
                        self._stage_tokens_since_refresh[stage] = 0
            self._current_single_stage = stage
            self._current_single_observe = observe
        return self._current_single_stage, self._current_single_observe

    def observe_or_mask(self, layer_id: int, intermediate: torch.Tensor) -> torch.Tensor | None:
        token_count = int(intermediate.shape[1])
        if token_count > 1:
            self.observe_prompt(layer_id, intermediate)
            return None
        stage, observe = self._single_token_mode(layer_id, token_count)
        if self.fallback_reason is not None or stage not in STAGES:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage["dense"] += token_count
            return None
        ratio = self.stage_ratios[stage]
        if ratio <= 0.0:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            return None
        self._append_recent(stage, layer_id, intermediate)
        if observe:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            return None
        if layer_id == 0:
            self.masked_tokens_by_stage[stage] += token_count
            self._stage_tokens_since_refresh[stage] += token_count
        return self.keep_mask(stage, layer_id)

    def keep_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        key = (stage, layer_id)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        ratio = self.stage_ratios[stage]
        base_entry = self.bank["policies"][self.base_policy][stage][layer_id]
        prior_entry = self.bank["policies"][self.stage_prior_policy][stage][layer_id]
        base_mask = base_entry["masks"][ratio_key(ratio)].bool()
        base_metric = base_entry["metric"].float()
        stage_prior = prior_entry["metric"].float()
        runtime_score = self._recent_sums[stage].get(layer_id)
        if runtime_score is None:
            runtime_score = self._prompt_scores.get(layer_id)
        if runtime_score is None:
            runtime_score = torch.ones_like(stage_prior)
        runtime_score = runtime_score.to(device=stage_prior.device, dtype=torch.float32).sqrt()
        channels = int(base_mask.numel())
        swap_count = min(
            channels - 1,
            max(0, int(round(channels * self.swap_ratios[stage]))),
        )
        mask = base_mask.clone().to(device=stage_prior.device)
        if swap_count:
            core_keep = int(round(int(mask.sum().item()) * self.static_core_ratios[stage]))
            protected_static = _topk_within_mask(base_metric, mask, core_keep)
            drop_candidates = mask & ~protected_static
            add_candidates = ~mask
            swap_count = min(
                swap_count,
                int(drop_candidates.sum().item()),
                int(add_candidates.sum().item()),
            )
            if swap_count:
                add_score = self.prior_weight * _zscore(stage_prior) + self.runtime_weight * _zscore(runtime_score)
                drop_score = self.prior_weight * _zscore(-base_metric) + self.runtime_weight * _zscore(-runtime_score)
                drop_mask = _topk_within_mask(drop_score, drop_candidates, swap_count)
                add_mask = _topk_within_mask(add_score, add_candidates, swap_count)
                mask[drop_mask] = False
                mask[add_mask] = True
        self._mask_cache[key] = mask
        self._swap_channels_by_stage_layer[stage][layer_id] = int(
            (mask ^ base_mask.to(device=mask.device)).sum().item() // 2
        )
        if layer_id == 0:
            self.mask_refresh_count_by_stage[stage] += 1
        return mask

    def entry(self, stage: str, layer_id: int) -> dict[str, Any]:
        return self.bank["policies"][self.stage_prior_policy][stage][layer_id]

    def summary(self) -> dict[str, Any]:
        total = sum(self.dense_observation_tokens_by_stage.values()) + sum(
            self.masked_tokens_by_stage.values()
        )
        weighted = sum(
            self.stage_ratios.get(stage, 0.0) * count
            for stage, count in self.masked_tokens_by_stage.items()
        )
        return {
            "backend": "calibrated_stage_static_core_residual_griffin_logical_v1",
            "real_speedup_claimed": False,
            "policy": "calibrated_stage_static_core_residual_griffin",
            "score_mode": "static_core_residual_swap",
            "alpha": self.alpha,
            "runtime_weight": self.runtime_weight,
            "prior_weight": self.prior_weight,
            "base_policy": self.base_policy,
            "stage_prior_policy": self.stage_prior_policy,
            "warmup_tokens": self.warmup_tokens,
            "stage_ratios": self.stage_ratios,
            "static_core_ratios": self.static_core_ratios,
            "swap_ratios": self.swap_ratios,
            "refresh_intervals": self.refresh_intervals,
            "window_tokens": self.window_tokens,
            "actual_swapped_channels_by_stage_layer": {
                stage: {str(layer_id): count for layer_id, count in values.items()}
                for stage, values in self._swap_channels_by_stage_layer.items()
                if values
            },
            "tokens_by_stage": dict(self.tokens_by_stage),
            "dense_observation_tokens_by_stage": dict(self.dense_observation_tokens_by_stage),
            "masked_tokens_by_stage": dict(self.masked_tokens_by_stage),
            "mask_refresh_count_by_stage": dict(self.mask_refresh_count_by_stage),
            "active_stage": self.active_stage,
            "fallback_reason": self.fallback_reason,
            "theoretical_average_mlp_pruning_ratio": weighted / total if total else 0.0,
        }


class AdaptiveStageGriffinQwen3MLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, layer_id: int, runtime: AdaptiveStageGriffinRuntime) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.layer_id = int(layer_id)
        self.runtime = runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        mask = self.runtime.observe_or_mask(self.layer_id, intermediate)
        if mask is None:
            return self.down_proj(intermediate)
        stage = self.runtime.active_stage
        if stage not in STAGES:
            return self.down_proj(intermediate)
        mask = mask.to(device=intermediate.device, dtype=intermediate.dtype)
        output = self.down_proj(intermediate * mask)
        if self.runtime.bias_compensation:
            entry = self.runtime.entry(stage, self.layer_id)
            removed_mean = entry["mean"].to(
                device=intermediate.device, dtype=intermediate.dtype
            ) * (1.0 - mask)
            output = output + removed_mean @ self.down_proj.weight.t()
        return output


class AttentionHeadMaskedOProj(nn.Module):
    def __init__(
        self,
        original_o_proj: nn.Module,
        layer_id: int,
        runtime: SafeDynamicStageGriffinRuntime,
        *,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.original_o_proj = original_o_proj
        self.layer_id = int(layer_id)
        self.runtime = runtime
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.original_o_proj.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.original_o_proj, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.runtime.observe_or_mask_attention(self.layer_id, x)
        if mask is None:
            return self.original_o_proj(x)
        if x.shape[-1] != self.num_heads * self.head_dim:
            raise ValueError(
                f"Attention hidden width mismatch in layer {self.layer_id}: "
                f"{x.shape[-1]} vs {self.num_heads * self.head_dim}"
            )
        head_mask = mask.to(device=x.device, dtype=x.dtype).reshape(
            *([1] * (x.ndim - 1)), self.num_heads, 1
        )
        values = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        return self.original_o_proj((values * head_mask).reshape_as(x))


def apply_adaptive_stage_griffin_qwen3(
    model: nn.Module,
    runtime: AdaptiveStageGriffinRuntime,
) -> nn.Module:
    layers = get_decoder_layers(model)
    if len(layers) != len(runtime.bank["layers"]):
        raise ValueError("Mask bank layer count does not match model")
    if isinstance(runtime, SafeDynamicStageGriffinRuntime) and runtime.score_mode == "output_aware":
        output_norms: dict[int, torch.Tensor] = {}
        for layer_id, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if not hasattr(mlp, "down_proj"):
                raise ValueError("Expected down_proj while configuring output-aware safe-dynamic scoring")
            output_norms[layer_id] = mlp.down_proj.weight.detach().float().square().sum(dim=0)
        runtime.set_output_norms(output_norms)
    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, AdaptiveStageGriffinQwen3MLP):
            if mlp.runtime is not runtime:
                mlp.runtime = runtime
            continue
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected gated MLP with gate_proj/up_proj/down_proj/act_fn")
        layer.mlp = AdaptiveStageGriffinQwen3MLP(mlp, layer_id, runtime)
    if isinstance(runtime, SafeDynamicStageGriffinRuntime) and runtime.attention_head_pruning_enabled:
        for layer_id, layer in enumerate(layers):
            attn = _find_self_attention(layer)
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise ValueError("Expected self_attn.o_proj for attention head pruning")
            if isinstance(o_proj, AttentionHeadMaskedOProj):
                runtime.register_attention_layer(
                    layer_id,
                    num_heads=o_proj.num_heads,
                    head_dim=o_proj.head_dim,
                    o_proj_weight=o_proj.weight,
                )
                if o_proj.runtime is not runtime:
                    o_proj.runtime = runtime
                continue
            hidden_size = int(o_proj.weight.shape[1])
            num_heads = _attention_num_heads(attn, hidden_size)
            head_dim = hidden_size // num_heads
            runtime.register_attention_layer(
                layer_id,
                num_heads=num_heads,
                head_dim=head_dim,
                o_proj_weight=o_proj.weight,
            )
            attn.o_proj = AttentionHeadMaskedOProj(
                o_proj,
                layer_id,
                runtime,
                num_heads=num_heads,
                head_dim=head_dim,
            )
    return model


apply_fixed_stage_masking = apply_fixed_stage_masking_qwen3
apply_griffin_prompt = apply_griffin_prompt_qwen3
apply_adaptive_stage_griffin = apply_adaptive_stage_griffin_qwen3
