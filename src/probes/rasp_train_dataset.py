from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.probes.action_conditioned_dataset import DATASETS, _feature_dataset, _one_hot
from src.utils.io import read_jsonl


DEFAULT_RATIOS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]


def ratio_index(ratio: float, ratios: list[float]) -> int:
    for index, candidate in enumerate(ratios):
        if abs(float(ratio) - float(candidate)) < 1e-8:
            return index
    raise ValueError(f"ratio {ratio} is not in supported ratios {ratios}")


def build_policy_features(
    hidden_state: torch.Tensor,
    *,
    entropy: float,
    confidence: float,
    position: float,
    target_budget: float,
    available_budget: float,
    dataset: str,
) -> torch.Tensor:
    dense_features = torch.tensor(
        [
            float(entropy),
            float(confidence),
            float(position),
            float(target_budget),
            float(available_budget),
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [
            hidden_state.float().flatten(),
            dense_features,
            _one_hot(_feature_dataset(dataset), DATASETS),
        ]
    )


class RaspTrainPolicyDataset(Dataset):
    """One example per reasoning step and target budget for ratio-policy imitation."""

    def __init__(
        self,
        jsonl_path: str | Path,
        hidden_path: str | Path,
        *,
        ratios: list[float] | None = None,
    ) -> None:
        self.rows = read_jsonl(jsonl_path)
        self.hidden = torch.load(hidden_path, map_location="cpu")
        self.ratios = [float(value) for value in (ratios or DEFAULT_RATIOS)]
        if len(self.rows) != len(self.hidden):
            raise ValueError(f"rows ({len(self.rows)}) and hidden states ({len(self.hidden)}) differ")
        for row in self.rows:
            candidate_flipped = row.get("candidate_flipped", [])
            candidate_unsafe = row.get("candidate_unsafe", candidate_flipped)
            if len(candidate_flipped) != len(self.ratios):
                raise ValueError("candidate_flipped length does not match supported ratios")
            if len(candidate_unsafe) != len(self.ratios):
                raise ValueError("candidate_unsafe length does not match supported ratios")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row: dict[str, Any] = self.rows[index]
        features = build_policy_features(
            self.hidden[index],
            entropy=float(row.get("entropy", 0.0)),
            confidence=float(row.get("confidence", 0.0)),
            position=float(row.get("position", 0.0)),
            target_budget=float(row["target_budget"]),
            available_budget=float(row.get("available_budget_before_selection", row["target_budget"])),
            dataset=str(row.get("dataset") or "unknown"),
        )
        label = ratio_index(float(row["oracle_ratio"]), self.ratios)
        unsafe_values = row.get("candidate_unsafe", row["candidate_flipped"])
        unsafe_mask = torch.tensor([float(value) for value in unsafe_values], dtype=torch.float32)
        ratios = torch.tensor(self.ratios, dtype=torch.float32)
        target_budget = torch.tensor(float(row["target_budget"]), dtype=torch.float32)
        return features, torch.tensor(label, dtype=torch.long), unsafe_mask, ratios, target_budget, index
