from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class HiddenStateRiskDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, hidden_path: str | Path):
        self.rows = read_jsonl(jsonl_path)
        self.hidden = torch.load(hidden_path, map_location="cpu")
        if len(self.rows) != len(self.hidden):
            raise ValueError(f"rows ({len(self.rows)}) and hidden states ({len(self.hidden)}) differ")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return self.hidden[idx].float(), torch.tensor(float(row["flipped"]), dtype=torch.float32), idx
