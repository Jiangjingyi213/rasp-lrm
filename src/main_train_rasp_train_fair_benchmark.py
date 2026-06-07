from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.rasp.fair_benchmark import (
    FAIR_BENCHMARK_SCHEMA,
    LABEL_FIELDS,
    VARIANTS,
    FairActionRiskDataset,
    build_model,
    calibrate_thresholds,
    indices_for_split,
    predict_risks,
)
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, positive_weight: float) -> dict:
    model.eval()
    labels, scores = [], []
    total_loss = 0.0
    total = 0
    weight = torch.tensor(positive_weight, device=device)
    for features, targets, ratios, _indices in loader:
        features, targets, ratios = features.to(device), targets.to(device), ratios.to(device)
        logits = model(features, ratios)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
        total_loss += float(loss.item()) * features.shape[0]
        total += features.shape[0]
        nonzero = ratios > 0
        labels.extend(int(value) for value in targets[nonzero].cpu().tolist())
        scores.extend(float(value) for value in torch.sigmoid(logits)[nonzero].cpu().tolist())
    return {
        "loss": total_loss / max(1, total),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--label-type", choices=sorted(LABEL_FIELDS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--max-flip-rates", nargs="+", type=float, default=[0.06, 0.08])
    parser.add_argument("--max-unsafe-rates", nargs="+", type=float, default=[0.08, 0.10])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not len(args.budgets) == len(args.max_flip_rates) == len(args.max_unsafe_rates):
        raise ValueError("Calibration budget and safety lists must have equal lengths")

    set_seed(args.seed)
    feature_set, model_type = VARIANTS[args.variant]
    dataset = FairActionRiskDataset(
        args.dataset, args.hidden_states, feature_set=feature_set, label_type=args.label_type
    )
    manifest = read_json(args.manifest)
    if manifest.get("schema") != FAIR_BENCHMARK_SCHEMA or int(manifest["seed"]) != args.seed:
        raise ValueError("Fair benchmark manifest schema or seed mismatch")
    train_indices = indices_for_split(dataset.rows, manifest, "train")
    calibration_indices = indices_for_split(dataset.rows, manifest, "calibration")
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    calibration_loader = DataLoader(Subset(dataset, calibration_indices), batch_size=args.batch_size)
    sample = dataset[0][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, int(sample.numel()), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    train_targets = torch.stack([dataset[index][1] for index in train_indices])
    positives = float(train_targets.sum())
    positive_weight = float(train_targets.numel() - positives) / max(1.0, positives)
    best = {"loss": float("inf")}
    output = Path(args.output)
    ensure_dir(output.parent)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, targets, ratios, _indices in train_loader:
            features, targets, ratios = features.to(device), targets.to(device), ratios.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, ratios)
            weight = torch.tensor(positive_weight, device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, calibration_loader, device, positive_weight)
        if metrics["loss"] < best["loss"]:
            best = {**metrics, "epoch": epoch}
            torch.save({"model": model.state_dict()}, output)
        print(f"epoch={epoch:02d} calibration_loss={metrics['loss']:.4f} roc_auc={metrics['roc_auc']}")

    model.load_state_dict(torch.load(output, map_location=device)["model"])
    cal_rows = [dataset.rows[index] for index in calibration_indices]
    cal_hidden = dataset.hidden[torch.tensor(calibration_indices)]
    risks = predict_risks(model, cal_rows, cal_hidden, dataset.ratios, feature_set, device)
    calibration = calibrate_thresholds(
        cal_rows, risks, dataset.ratios, args.budgets, args.max_flip_rates, args.max_unsafe_rates
    )
    metadata = {
        "schema": FAIR_BENCHMARK_SCHEMA,
        "variant": args.variant,
        "feature_set": feature_set,
        "model_type": model_type,
        "label_type": args.label_type,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "dim": int(sample.numel()),
        "hidden_dim": args.hidden_dim,
        "ratios": dataset.ratios,
        "positive_weight": positive_weight,
        "best": best,
        "calibration": calibration,
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)
    write_json(args.metrics_output, metadata)


if __name__ == "__main__":
    main()
