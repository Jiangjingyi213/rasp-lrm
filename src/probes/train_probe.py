from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.probes.dataset import HiddenStateRiskDataset
from src.utils.io import ensure_dir


@dataclass
class ProbeResult:
    roc_auc: float | None
    pr_auc: float | None
    val_loss: float


class LinearRiskProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> ProbeResult:
    model.eval()
    labels, scores = [], []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y, _rows in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(loss_fn(logits, y).item()) * len(y)
            labels.extend(y.cpu().int().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    if len(set(labels)) < 2:
        roc_auc = pr_auc = None
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    return ProbeResult(roc_auc=roc_auc, pr_auc=pr_auc, val_loss=total_loss / max(len(labels), 1))


def train_probe(
    jsonl_path: str,
    hidden_path: str,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 1,
    split: str = "problem",
) -> dict:
    dataset = HiddenStateRiskDataset(jsonl_path, hidden_path)
    if len(dataset) < 2:
        raise ValueError("Need at least two probe examples to train/validate")
    if split == "problem":
        train_indices, val_indices = problem_level_split(dataset.rows, val_fraction=val_fraction, seed=seed)
    elif split == "row":
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        val_size = max(1, int(len(dataset) * val_fraction))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
    else:
        raise ValueError(f"Unsupported probe split: {split}")
    train_ds, val_ds = Subset(dataset, train_indices), Subset(dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = dataset.hidden.shape[-1]
    model = LinearRiskProbe(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    best = {"roc_auc": None, "pr_auc": None, "val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, _rows in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        result = evaluate(model, val_loader, device)
        if result.val_loss < best["val_loss"]:
            best = {**result.__dict__, "epoch": epoch, **split_summary(dataset.rows, train_indices, val_indices, split)}
            ensure_dir(Path(output_path).parent)
            torch.save({"model": model.state_dict(), "dim": dim, "best": best}, output_path)
    return best


def problem_level_split(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    by_id: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_id.setdefault(str(row["id"]), []).append(idx)
    ids = list(by_id)
    random.Random(seed).shuffle(ids)
    val_problem_count = max(1, int(len(ids) * val_fraction))
    if val_problem_count >= len(ids) and len(ids) > 1:
        val_problem_count = len(ids) - 1
    val_ids = set(ids[:val_problem_count])
    train_indices, val_indices = [], []
    for item_id, indices in by_id.items():
        if item_id in val_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    if not train_indices or not val_indices:
        raise ValueError("Problem-level split produced an empty train or validation split")
    return train_indices, val_indices


def split_summary(rows: list[dict], train_indices: list[int], val_indices: list[int], split: str) -> dict:
    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]
    return {
        "split": split,
        "train_problem_count": len({str(row["id"]) for row in train_rows}),
        "val_problem_count": len({str(row["id"]) for row in val_rows}),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "positive_rate_train": sum(int(row["flipped"]) for row in train_rows) / len(train_rows),
        "positive_rate_val": sum(int(row["flipped"]) for row in val_rows) / len(val_rows),
    }
