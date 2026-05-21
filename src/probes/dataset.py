from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class HiddenStateRiskDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        hidden_path: str | Path | None = None,
        activation_path: str | Path | None = None,
        feature_set: str = "hidden",
    ):
        self.rows = read_jsonl(jsonl_path)
        self.feature_set = feature_set
        self.hidden = torch.load(hidden_path, map_location="cpu") if hidden_path else None
        self.activation = torch.load(activation_path, map_location="cpu") if activation_path else None
        if self.hidden is not None and len(self.rows) != len(self.hidden):
            raise ValueError(f"rows ({len(self.rows)}) and hidden states ({len(self.hidden)}) differ")
        if self.activation is not None and len(self.rows) != len(self.activation):
            raise ValueError(f"rows ({len(self.rows)}) and activation features ({len(self.activation)}) differ")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        features = []
        if self.feature_set in {"hidden", "combined"}:
            if self.hidden is None:
                raise ValueError("hidden features requested but hidden_path was not provided")
            features.append(self.hidden[idx].float())
        if self.feature_set in {"activation", "combined"}:
            if self.activation is None:
                raise ValueError("activation features requested but activation_path was not provided")
            features.append(self.activation[idx].float())
        if self.feature_set in {"entropy", "combined"}:
            features.append(torch.tensor([float(row.get("entropy", 0.0))], dtype=torch.float32))
        if self.feature_set in {"confidence", "combined"}:
            features.append(torch.tensor([float(row.get("confidence", 0.0))], dtype=torch.float32))
        if not features:
            raise ValueError(f"Unsupported feature set: {self.feature_set}")
        return torch.cat([feature.flatten() for feature in features]), torch.tensor(float(row["flipped"]), dtype=torch.float32), idx
