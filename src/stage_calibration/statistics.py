from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import STAGES


@dataclass
class TokenMoments:
    count: int
    sum: torch.Tensor
    sum_sq: torch.Tensor

    @classmethod
    def zeros(cls, channels: int) -> "TokenMoments":
        return cls(0, torch.zeros(channels, dtype=torch.float64), torch.zeros(channels, dtype=torch.float64))

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float64, device="cpu")
        self.count += int(values.shape[0])
        self.sum += values.sum(dim=0)
        self.sum_sq += values.square().sum(dim=0)

    def mean(self) -> torch.Tensor:
        return self.sum / max(1, self.count)

    def variance(self) -> torch.Tensor:
        if self.count <= 1:
            return torch.zeros_like(self.sum)
        mean = self.mean()
        return (self.sum_sq / self.count - mean.square()).clamp_min(0.0)


def wifv(moment: TokenMoments, down_proj_weight: torch.Tensor) -> torch.Tensor:
    weight_norm_sq = down_proj_weight.detach().float().cpu().square().sum(dim=0)
    return moment.variance().float() * weight_norm_sq


def keep_mask(values: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    channels = int(values.numel())
    prune = int(round(channels * float(ratio)))
    prune = min(channels - 1, max(0, prune))
    mask = torch.ones(channels, dtype=torch.bool)
    if prune:
        indices = torch.topk(values.float(), k=prune, largest=False).indices
        mask[indices] = False
    return mask


def stage_balanced_metric(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    missing = [stage for stage in STAGES if stage not in metrics]
    if missing:
        raise ValueError(f"Missing stage metrics: {missing}")
    normalized = []
    for stage in STAGES:
        values = metrics[stage].float()
        normalized.append((values - values.mean()) / values.std().clamp_min(1e-12))
    return torch.stack(normalized).mean(dim=0)
