from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn

from src.models.hooks import get_decoder_layers

from .mask_bank import ratio_key, validate_mask_bank
from .protocol import STAGES


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
