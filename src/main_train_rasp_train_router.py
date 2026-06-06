from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.probes.rasp_train_dataset import RaspTrainPolicyDataset
from src.probes.train_probe import problem_level_split
from src.rasp.train_policy import RatioPolicyNet, checkpoint_metadata, evaluate_ratio_policy, ratio_policy_loss
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def policy_split_summary(
    rows: list[dict],
    train_indices: list[int],
    val_indices: list[int],
) -> dict:
    train_rows = [rows[index] for index in train_indices]
    val_rows = [rows[index] for index in val_indices]

    def unsafe_rate(items: list[dict]) -> float:
        values = [bool(value) for row in items for value in row["candidate_unsafe"]]
        return sum(int(value) for value in values) / max(1, len(values))

    return {
        "split": "problem",
        "train_problem_count": len({(str(row.get("dataset") or "unknown"), str(row["id"])) for row in train_rows}),
        "val_problem_count": len({(str(row.get("dataset") or "unknown"), str(row["id"])) for row in val_rows}),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unsafe_candidate_rate_train": unsafe_rate(train_rows),
        "unsafe_candidate_rate_val": unsafe_rate(val_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--unsafe-weight", type=float, default=3.0)
    parser.add_argument("--budget-weight", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = RaspTrainPolicyDataset(args.dataset, args.hidden_states)
    train_indices, val_indices = problem_level_split(dataset.rows, args.val_fraction, args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample, _, _, ratios, _, _ = dataset[0]
    model = RatioPolicyNet(int(sample.shape[-1]), int(ratios.shape[-1]), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    output = Path(args.output)
    ensure_dir(output.parent)
    best = {"val_loss": float("inf")}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, labels, unsafe_mask, batch_ratios, target_budget, _indices in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            unsafe_mask = unsafe_mask.to(device)
            batch_ratios = batch_ratios.to(device)
            target_budget = target_budget.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ratio_policy_loss(
                logits,
                labels,
                unsafe_mask,
                batch_ratios,
                target_budget,
                unsafe_weight=args.unsafe_weight,
                budget_weight=args.budget_weight,
            )
            loss.backward()
            optimizer.step()
        result = evaluate_ratio_policy(
            model,
            val_loader,
            device,
            unsafe_weight=args.unsafe_weight,
            budget_weight=args.budget_weight,
        )
        if result.val_loss < float(best["val_loss"]):
            best = {
                **result.__dict__,
                "epoch": epoch,
                "feature_set": "rasp_train_policy_hidden",
                "unsafe_weight": float(args.unsafe_weight),
                "budget_weight": float(args.budget_weight),
                **policy_split_summary(dataset.rows, train_indices, val_indices),
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "dim": int(sample.shape[-1]),
                    "num_ratios": int(ratios.shape[-1]),
                    "hidden_dim": int(args.hidden_dim),
                    "ratios": [float(value) for value in dataset.ratios],
                    "best": best,
                    "metadata": checkpoint_metadata(
                        ratios=dataset.ratios,
                        feature_set="rasp_train_policy_hidden",
                        best=best,
                    ),
                },
                output,
            )
    write_json(args.metrics_output, best)


if __name__ == "__main__":
    main()
