from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.rasp.phase_b2 import PhaseB2Dataset, calibrate_problem_folds, indices_for_split, validate_phase_b2_manifest
from src.rasp.phase_b25 import (
    PHASE_B25_SCHEMA,
    PHASE_B25_VARIANTS,
    PhaseB25ActionNet,
    boundary_any_flip_metrics,
    fit_phase_b25_transform,
    predict_phase_b25,
    transform_phase_b25_features,
)
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_seed


def metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=sorted(PHASE_B25_VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--max-flip-rates", nargs="+", type=float, default=[0.04, 0.06])
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if len(args.budgets) != len(args.max_flip_rates):
        raise ValueError("Budgets and max flip rates must have matching lengths")
    set_seed(args.seed)
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, "ratio_only")
    manifest = read_json(args.manifest)
    validate_phase_b2_manifest(dataset.rows, manifest, args.seed)
    split_indices = {
        name: indices_for_split(dataset.rows, manifest, name)
        for name in ("train", "validation", "calibration")
    }
    transform = fit_phase_b25_transform(
        dataset.rows, dataset.hidden, split_indices["train"], pca_dim=args.pca_dim
    )
    uncertainty, hidden_pca = transform_phase_b25_features(dataset.rows, dataset.hidden, transform)
    ratios = dataset.rows[0]["candidate_ratios"]
    targets = torch.tensor([row["candidate_flipped"] for row in dataset.rows], dtype=torch.float32)
    train = torch.tensor(split_indices["train"], dtype=torch.long)
    validation = torch.tensor(split_indices["validation"], dtype=torch.long)
    positives = float(targets[train, 1:].sum())
    positive_weight = float(targets[train, 1:].numel() - positives) / max(1.0, positives)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseB25ActionNet(args.variant, transform["pca_dim"], args.model_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
    nonzero = ratio_tensor > 0
    best = {"loss": float("inf")}
    output = Path(args.output)
    ensure_dir(output.parent)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(uncertainty[train].to(device), hidden_pca[train].to(device), ratio_tensor)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, nonzero], targets[train][:, 1:].to(device),
            pos_weight=torch.tensor(positive_weight, device=device),
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                uncertainty[validation].to(device), hidden_pca[validation].to(device), ratio_tensor
            )
            validation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                validation_logits[:, nonzero], targets[validation][:, 1:].to(device),
                pos_weight=torch.tensor(positive_weight, device=device),
            )
        if float(validation_loss) < best["loss"]:
            rows = [dataset.rows[index] for index in split_indices["validation"]]
            risks = predict_phase_b25(
                model, uncertainty[validation], hidden_pca[validation], ratios, device
            )
            action_labels = [int(value) for row in rows for value in row["candidate_flipped"][1:]]
            action_scores = [float(value) for row in risks for value in row[1:]]
            boundary_labels, boundary_scores = boundary_any_flip_metrics(rows, risks)
            best = {
                "loss": float(validation_loss),
                "epoch": epoch,
                "action": metrics(action_labels, action_scores),
                "boundary_any_flip": metrics(boundary_labels, boundary_scores),
            }
            torch.save({"model": model.state_dict()}, output)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_loss={float(loss.detach().item()):.4f} "
                f"validation_loss={float(validation_loss.item()):.4f} best_epoch={best.get('epoch')}"
            )
    model.load_state_dict(torch.load(output, map_location=device)["model"])
    calibration_indices = split_indices["calibration"]
    cal_rows = [dataset.rows[index] for index in calibration_indices]
    cal_tensor = torch.tensor(calibration_indices, dtype=torch.long)
    cal_risks = predict_phase_b25(
        model, uncertainty[cal_tensor], hidden_pca[cal_tensor], ratios, device
    )
    calibration = calibrate_problem_folds(
        cal_rows, cal_risks, ratios, budgets=args.budgets, max_flip_rates=args.max_flip_rates,
        folds=args.calibration_folds, seed=args.seed,
    )
    metadata = {
        "schema": PHASE_B25_SCHEMA,
        "variant": args.variant,
        "seed": args.seed,
        "pca_dim": transform["pca_dim"],
        "model_dim": args.model_dim,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "transform_fit_split": transform["fit_split"],
        "transform_fit_row_count": transform["fit_row_count"],
        "checkpoint_selection_split": "validation",
        "split_problem_counts": manifest["problem_counts"],
        "positive_weight": positive_weight,
        "best": best,
        "calibration": calibration,
        "ratios": ratios,
    }
    torch.save({"model": model.state_dict(), "metadata": metadata, "transform": transform}, output)
    write_json(args.metrics_output, metadata)


if __name__ == "__main__":
    main()
