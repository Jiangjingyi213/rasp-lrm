from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.rasp.phase_b2 import (
    PHASE_B2_SCHEMA,
    PHASE_B2_VARIANTS,
    PhaseB2Dataset,
    PhaseB2MultiTaskNet,
    calibrate_problem_folds,
    indices_for_split,
    multitask_loss,
    predict_phase_b2,
)
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device, positive_weight, divergence_weight, hidden_drift_weight) -> dict:
    model.eval()
    total_loss, total = 0.0, 0
    labels, scores = [], []
    divergence_error, drift_error, auxiliary_count = 0.0, 0.0, 0
    for features, flipped, divergence, drift, ratios, _indices in loader:
        features, flipped, divergence, drift, ratios = (
            features.to(device), flipped.to(device), divergence.to(device), drift.to(device), ratios.to(device)
        )
        outputs = model(features, ratios)
        loss, _parts = multitask_loss(
            outputs, flipped, divergence, drift, ratios,
            positive_weight=positive_weight,
            divergence_weight=divergence_weight,
            hidden_drift_weight=hidden_drift_weight,
        )
        total_loss += float(loss.item()) * features.shape[0]
        total += features.shape[0]
        nonzero = ratios > 0
        labels.extend(int(value) for value in flipped[nonzero].cpu().tolist())
        scores.extend(float(value) for value in torch.sigmoid(outputs["flip_logits"])[nonzero].cpu().tolist())
        divergence_error += float(torch.abs(outputs["token_divergence"][nonzero] - divergence[nonzero]).sum())
        drift_error += float(torch.abs(outputs["hidden_drift"][nonzero] - drift[nonzero]).sum())
        auxiliary_count += int(nonzero.sum())
    return {
        "loss": total_loss / max(1, total),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None,
        "token_divergence_mae": divergence_error / max(1, auxiliary_count),
        "hidden_drift_mae": drift_error / max(1, auxiliary_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=sorted(PHASE_B2_VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--divergence-weight", type=float, default=0.5)
    parser.add_argument("--hidden-drift-weight", type=float, default=0.5)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--max-flip-rates", nargs="+", type=float, default=[0.04, 0.06])
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if len(args.budgets) != len(args.max_flip_rates):
        raise ValueError("Budgets and max flip rates must have matching lengths")
    set_seed(args.seed)
    feature_set, multitask = PHASE_B2_VARIANTS[args.variant]
    divergence_weight = args.divergence_weight if multitask else 0.0
    hidden_drift_weight = args.hidden_drift_weight if multitask else 0.0
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, feature_set)
    manifest = read_json(args.manifest)
    if manifest.get("schema") != PHASE_B2_SCHEMA or int(manifest["seed"]) != args.seed:
        raise ValueError("Phase B2 manifest schema or seed mismatch")
    train_indices = indices_for_split(dataset.rows, manifest, "train")
    calibration_indices = indices_for_split(dataset.rows, manifest, "calibration")
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    calibration_loader = DataLoader(Subset(dataset, calibration_indices), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = int(dataset[0][0].numel())
    model = PhaseB2MultiTaskNet(dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    train_flips = torch.stack([dataset[index][1] for index in train_indices])[:, 1:]
    positives = float(train_flips.sum())
    positive_weight = float(train_flips.numel() - positives) / max(1.0, positives)
    output = Path(args.output)
    ensure_dir(output.parent)
    best = {"loss": float("inf")}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, flipped, divergence, drift, ratios, _indices in train_loader:
            features, flipped, divergence, drift, ratios = (
                features.to(device), flipped.to(device), divergence.to(device), drift.to(device), ratios.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, ratios)
            loss, _parts = multitask_loss(
                outputs, flipped, divergence, drift, ratios,
                positive_weight=positive_weight,
                divergence_weight=divergence_weight,
                hidden_drift_weight=hidden_drift_weight,
            )
            loss.backward()
            optimizer.step()
        metrics = evaluate(
            model, calibration_loader, device, positive_weight, divergence_weight, hidden_drift_weight
        )
        if metrics["loss"] < best["loss"]:
            best = {**metrics, "epoch": epoch}
            torch.save({"model": model.state_dict()}, output)
        print(f"epoch={epoch:02d} loss={metrics['loss']:.4f} auc={metrics['roc_auc']} pr={metrics['pr_auc']}")
    model.load_state_dict(torch.load(output, map_location=device)["model"])
    cal_rows = [dataset.rows[index] for index in calibration_indices]
    cal_hidden = dataset.hidden[torch.tensor(calibration_indices)]
    predictions = predict_phase_b2(model, cal_rows, cal_hidden, cal_rows[0]["candidate_ratios"], feature_set, device)
    calibration = calibrate_problem_folds(
        cal_rows, predictions["risks"], cal_rows[0]["candidate_ratios"],
        budgets=args.budgets,
        max_flip_rates=args.max_flip_rates,
        folds=args.calibration_folds,
        seed=args.seed,
    )
    metadata = {
        "schema": PHASE_B2_SCHEMA,
        "variant": args.variant,
        "feature_set": feature_set,
        "multitask": multitask,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "dim": dim,
        "hidden_dim": args.hidden_dim,
        "ratios": cal_rows[0]["candidate_ratios"],
        "positive_weight": positive_weight,
        "divergence_weight": divergence_weight,
        "hidden_drift_weight": hidden_drift_weight,
        "best": best,
        "calibration": calibration,
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)
    write_json(args.metrics_output, metadata)


if __name__ == "__main__":
    main()
