from __future__ import annotations

from collections import Counter
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


class StageMaskRuntime:
    def __init__(self, bank: dict[str, Any], policy: str, stage_ratios: dict[str, float], bias_compensation: bool = True) -> None:
        validate_mask_bank(bank)
        if policy not in bank["policies"]:
            raise ValueError(f"Unknown mask policy: {policy}")
        self.bank = bank
        self.policy = policy
        self.stage_ratios = {stage: float(stage_ratios.get(stage, 0.0)) for stage in STAGES}
        allowed = {float(value) for value in bank["ratios"]}
        if any(value not in allowed for value in self.stage_ratios.values()):
            raise ValueError("Stage ratio is not present in mask bank")
        self.bias_compensation = bool(bias_compensation)
        self.active_stage: str | None = None
        self.fallback_reason: str | None = None
        self.tokens_by_stage: Counter[str] = Counter()

    def reset(self) -> None:
        self.active_stage = None
        self.fallback_reason = None
        self.tokens_by_stage.clear()

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

    def summary(self) -> dict[str, Any]:
        total = sum(self.tokens_by_stage.values())
        weighted = sum(
            self.stage_ratios.get(stage, 0.0) * count
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

    @property
    def intermediate_size(self) -> int:
        return int(self.down_proj.weight.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        entry = self.runtime.entry(self.layer_id)
        if entry is None:
            return self.down_proj(intermediate)
        ratio = self.runtime.active_ratio()
        mask = entry["masks"][ratio_key(ratio)].to(device=intermediate.device, dtype=intermediate.dtype)
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
        allowed = {float(value) for value in bank["ratios"]}
        if any(value not in allowed for value in self.stage_ratios.values()):
            raise ValueError("Adaptive stage ratio is not present in mask bank")
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
