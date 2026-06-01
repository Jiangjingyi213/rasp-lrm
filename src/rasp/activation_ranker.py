from __future__ import annotations

import torch


def rank_intermediate_neurons(intermediate_states: torch.Tensor) -> torch.Tensor:
    """Rank FFN intermediate neurons from the observed prefix activation.

    The score follows the sequence-level normalization used by the local
    GRIFFIN adapter: normalize each token activation, then accumulate its
    magnitude across the observed prefix. The returned indices are ordered
    from most to least important and therefore define nested keep masks.
    """

    if intermediate_states.ndim != 3:
        raise ValueError(
            "Expected FFN intermediate states with shape [batch, sequence, channels], "
            f"got {tuple(intermediate_states.shape)}"
        )
    if intermediate_states.shape[0] != 1:
        raise ValueError("Runtime RASP-Zero v0 currently supports batch size 1")
    normalized = intermediate_states / intermediate_states.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scores = normalized.norm(dim=1)
    return torch.argsort(scores[0], descending=True)


def keep_indices_from_ranking(ranking: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    if not 0.0 <= float(prune_ratio) < 1.0:
        raise ValueError(f"Pruning ratio must be in [0, 1), got {prune_ratio}")
    channels = int(ranking.numel())
    keep = max(1, channels - int(round(channels * float(prune_ratio))))
    return ranking[:keep]


def keep_mask_from_ranking(ranking: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    keep_indices = keep_indices_from_ranking(ranking, prune_ratio)
    mask = torch.zeros(ranking.numel(), dtype=torch.bool, device=ranking.device)
    mask[keep_indices] = True
    return mask
