from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.main_offline_baselines import MODULE_STRENGTH
from src.utils.io import read_jsonl


MODULES = ["attention_heads", "attention_block", "mlp_channels", "mlp_intermediate_channels", "mlp_block", "layer"]
STAGES = ["understanding", "planning", "derivation", "verification", "final", "unknown"]
DATASETS = ["gsm8k", "math_train", "math500", "unknown"]


def _one_hot(value: str, vocabulary: list[str]) -> torch.Tensor:
    out = torch.zeros(len(vocabulary), dtype=torch.float32)
    if value in vocabulary:
        out[vocabulary.index(value)] = 1.0
    return out


def _feature_dataset(dataset: str) -> str:
    # Preserve source provenance in rows, but share one math-domain router feature.
    return "math500" if dataset == "math_train" else dataset


def build_action_features(
    hidden_state: torch.Tensor,
    *,
    entropy: float,
    confidence: float,
    position: float,
    ratio: float,
    module: str,
    dataset: str,
    pruned_layers: list[int],
    layer_dim: int,
    stage: str = "unknown",
    include_stage: bool = False,
) -> torch.Tensor:
    layer_mask = torch.zeros(layer_dim, dtype=torch.float32)
    for layer in pruned_layers:
        if layer is not None:
            layer_mask[int(layer)] = 1.0
    features = [
        hidden_state.float().flatten(),
        torch.tensor(
            [
                float(entropy),
                float(confidence),
                float(position),
                float(ratio),
                MODULE_STRENGTH.get(module, 0.5) * float(ratio),
            ],
            dtype=torch.float32,
        ),
        _one_hot(module, MODULES),
        _one_hot(_feature_dataset(dataset), DATASETS),
        layer_mask,
    ]
    if include_stage:
        features.append(_one_hot(stage, STAGES))
    return torch.cat(features)


class ActionConditionedRiskDataset(Dataset):
    """Risk examples conditioned on reasoning state and a candidate pruning action."""

    def __init__(
        self,
        jsonl_path: str | Path,
        hidden_path: str | Path,
        include_stage: bool,
        layer_dim: int | None = None,
    ):
        self.rows = read_jsonl(jsonl_path)
        self.hidden = torch.load(hidden_path, map_location="cpu")
        self.include_stage = include_stage
        if len(self.rows) != len(self.hidden):
            raise ValueError(f"rows ({len(self.rows)}) and hidden states ({len(self.hidden)}) differ")
        inferred_layer_dim = max(
            [int(layer) for row in self.rows for layer in row.get("pruned_layers", []) if layer is not None] + [0]
        ) + 1
        self.layer_dim = max(inferred_layer_dim, int(layer_dim or 0))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        ratio = float(row.get("ratio", 0.0))
        module = str(row.get("module", "none"))
        num_segments = max(1, int(row.get("num_segments", 1)))
        segment_index = int(row.get("segment_index", row.get("segment_id", 0)))
        position = segment_index / max(1, num_segments - 1)
        features = build_action_features(
            self.hidden[index],
            entropy=float(row.get("entropy", 0.0)),
            confidence=float(row.get("confidence", 0.0)),
            position=position,
            ratio=ratio,
            module=module,
            dataset=str(row.get("dataset") or "unknown"),
            pruned_layers=[int(layer) for layer in row.get("pruned_layers", []) if layer is not None],
            layer_dim=self.layer_dim,
            stage=str(row.get("segment_type", "unknown")),
            include_stage=self.include_stage,
        )
        return features, torch.tensor(float(row["flipped"]), dtype=torch.float32), index
