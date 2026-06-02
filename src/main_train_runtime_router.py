from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.probes.action_conditioned_dataset import ActionConditionedRiskDataset
from src.probes.train_probe import LinearRiskProbe, evaluate, problem_level_split, split_summary
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = ActionConditionedRiskDataset(args.dataset, args.hidden_states, include_stage=False)
    train_indices, val_indices = problem_level_split(dataset.rows, args.val_fraction, args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample, _, _ = dataset[0]
    model = LinearRiskProbe(int(sample.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"val_loss": float("inf")}
    output = Path(args.output)
    ensure_dir(output.parent)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, _indices in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        result = evaluate(model, val_loader, device)
        if result.val_loss < float(best["val_loss"]):
            best = {
                **result.__dict__,
                "epoch": epoch,
                "feature_set": "action_hidden",
                **split_summary(dataset.rows, train_indices, val_indices, "problem"),
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "dim": int(sample.shape[-1]),
                    "best": best,
                    "metadata": {
                        "feature_set": "action_hidden",
                        "include_stage": False,
                        "layer_dim": dataset.layer_dim,
                        "module": "mlp_intermediate_channels",
                    },
                },
                output,
            )
    write_json(args.metrics_output, best)


if __name__ == "__main__":
    main()
