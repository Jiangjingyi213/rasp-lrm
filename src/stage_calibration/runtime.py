from __future__ import annotations

from collections import Counter, deque
from typing import Any

import torch
from torch import nn

from src.models.hooks import get_decoder_layers

from .mask_bank import ratio_key, validate_mask_bank
from .protocol import STAGES


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


class StageMaskRuntime:
    def __init__(self, bank: dict[str, Any], policy: str, stage_ratios: dict[str, float], bias_compensation: bool = True) -> None:
        validate_mask_bank(bank)
        if policy not in bank["policies"]:
            raise ValueError(f"Unknown mask policy: {policy}")
        self.bank = bank
        self.policy = policy
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        self.bias_compensation = bool(bias_compensation)
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
            raise ValueError("Expected Qwen3 MLP")
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
            raise ValueError("Expected Qwen3 MLP")
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
        if self.fallback_reason is not None or stage not in STAGES:
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
        ratio = self.stage_ratios[stage]
        if ratio <= 0.0:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            self._record_actual_pruning(stage, layer_id, token_count, None)
            return None
        self._append_recent(stage, layer_id, intermediate)
        if observe:
            if layer_id == 0:
                self.dense_observation_tokens_by_stage[stage] += token_count
            self._record_actual_pruning(stage, layer_id, token_count, None)
            return None
        if layer_id == 0:
            self.masked_tokens_by_stage[stage] += token_count
            self._stage_tokens_since_refresh[stage] += token_count
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
            runtime_score = self._prompt_scores.get(layer_id)
        if runtime_score is None:
            runtime_score = torch.ones_like(prior)
        runtime_score = runtime_score.to(device=prior.device, dtype=torch.float32).sqrt()
        protected = _topk_mask(prior, self.protected_core_ratios[stage])
        prune_score = self.prior_weight * _zscore(-prior) + self.runtime_weight * _zscore(-runtime_score)
        mask = _prune_mask_from_scores(
            prune_score,
            self.stage_ratios[stage],
            protected_mask=protected,
        )
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
        actual_by_stage = {
            stage: (
                float(self._actual_pruning_weighted_sum_by_stage[stage])
                / float(self._actual_pruning_denominator_by_stage[stage])
            )
            for stage in self._actual_pruning_denominator_by_stage
            if self._actual_pruning_denominator_by_stage[stage]
        }
        return {
            "backend": "calibrated_stage_safe_dynamic_griffin_logical_v2",
            "real_speedup_claimed": False,
            "policy": "calibrated_stage_safe_dynamic_griffin",
            "score_mode": "protected_core_safe_prune",
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
            "theoretical_average_mlp_pruning_ratio": weighted / total if total else 0.0,
            "actual_average_mlp_pruning_ratio": (
                self._actual_pruning_weighted_sum / self._actual_pruning_denominator
                if self._actual_pruning_denominator
                else 0.0
            ),
            "actual_pruning_ratio_by_stage": actual_by_stage,
            "actual_pruning_accounting": "actual_mask_sparsity_layer_token_weighted",
        }


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


def apply_adaptive_stage_griffin_qwen3(
    model: nn.Module,
    runtime: AdaptiveStageGriffinRuntime,
) -> nn.Module:
    layers = get_decoder_layers(model)
    if len(layers) != len(runtime.bank["layers"]):
        raise ValueError("Mask bank layer count does not match model")
    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, AdaptiveStageGriffinQwen3MLP):
            if mlp.runtime is not runtime:
                mlp.runtime = runtime
            continue
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected Qwen3 MLP")
        layer.mlp = AdaptiveStageGriffinQwen3MLP(mlp, layer_id, runtime)
    return model
