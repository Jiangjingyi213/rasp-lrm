from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, random_split

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
        for x, y in loader:
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
) -> dict:
    dataset = HiddenStateRiskDataset(jsonl_path, hidden_path)
    if len(dataset) < 2:
        raise ValueError("Need at least two probe examples to train/validate")
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
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
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        result = evaluate(model, val_loader, device)
        if result.val_loss < best["val_loss"]:
            best = {**result.__dict__, "epoch": epoch}
            ensure_dir(Path(output_path).parent)
            torch.save({"model": model.state_dict(), "dim": dim, "best": best}, output_path)
    return best
