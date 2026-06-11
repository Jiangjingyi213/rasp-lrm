from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.rasp.phase_b2 import PhaseB2Dataset, calibrate_problem_folds, indices_for_split, validate_phase_b2_manifest
from src.rasp.phase_b25 import fit_phase_b25_transform, transform_phase_b25_features
from src.rasp.phase_b25b import (
    PHASE_B25B_SCHEMA,
    HiddenActionResidual,
    combined_risks,
    frozen_uncertainty_logits,
)
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_seed


def weighted_loss(logits: torch.Tensor, targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:], targets[:, 1:],
        pos_weight=torch.tensor(positive_weight, device=logits.device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--max-flip-rates", nargs="+", type=float, default=[0.04, 0.06])
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if len(args.budgets) != len(args.max_flip_rates):
        raise ValueError("Budgets and max flip rates must have matching lengths")
    if 0.0 not in args.alphas:
        raise ValueError("Phase B2.5b alpha candidates must include zero")
    set_seed(args.seed)
    dataset = PhaseB2Dataset(args.dataset, args.hidden_states, "ratio_only")
    manifest = read_json(args.manifest)
    validate_phase_b2_manifest(dataset.rows, manifest, args.seed)
    splits = {
        name: indices_for_split(dataset.rows, manifest, name)
        for name in ("train", "validation", "calibration")
    }
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    if int(base_checkpoint["metadata"].get("seed", -1)) != args.seed:
        raise ValueError("Frozen uncertainty checkpoint seed does not match manifest")
    if base_checkpoint["metadata"].get("checkpoint_selection_split") != "validation":
        raise ValueError("Frozen uncertainty checkpoint must be selected on validation")
    transform = fit_phase_b25_transform(dataset.rows, dataset.hidden, splits["train"], pca_dim=args.pca_dim)
    _uncertainty, hidden_pca = transform_phase_b25_features(dataset.rows, dataset.hidden, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_logits = frozen_uncertainty_logits(base_checkpoint, dataset.rows, dataset.hidden, device).to(device)
    targets = torch.tensor([row["candidate_flipped"] for row in dataset.rows], dtype=torch.float32, device=device)
    train = torch.tensor(splits["train"], dtype=torch.long, device=device)
    validation = torch.tensor(splits["validation"], dtype=torch.long, device=device)
    positives = float(targets[train, 1:].sum())
    positive_weight = float(targets[train, 1:].numel() - positives) / max(1.0, positives)
    ratios = base_checkpoint["metadata"]["ratios"]
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)
    model = HiddenActionResidual(transform["pca_dim"], args.model_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output)
    ensure_dir(output.parent)
    best = {
        "loss": float(weighted_loss(base_logits[validation], targets[validation], positive_weight).item()),
        "epoch": 0,
        "alpha": 0.0,
    }
    torch.save({"model": model.state_dict()}, output)
    hidden_device = hidden_pca.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(hidden_device[train], ratio_tensor)
        loss = weighted_loss(base_logits[train] + residual, targets[train], positive_weight)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_residual = model(hidden_device[validation], ratio_tensor)
            for alpha in args.alphas:
                validation_loss = weighted_loss(
                    base_logits[validation] + float(alpha) * validation_residual,
                    targets[validation],
                    positive_weight,
                )
                if float(validation_loss.item()) < best["loss"]:
                    best = {"loss": float(validation_loss.item()), "epoch": epoch, "alpha": float(alpha)}
                    torch.save({"model": model.state_dict()}, output)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={float(loss.detach().item()):.4f} best={best}")
    model.load_state_dict(torch.load(output, map_location=device)["model"])
    calibration = torch.tensor(splits["calibration"], dtype=torch.long, device=device)
    with torch.no_grad():
        cal_residual = model(hidden_device[calibration], ratio_tensor).cpu()
    cal_rows = [dataset.rows[index] for index in splits["calibration"]]
    cal_risks = combined_risks(base_logits[calibration].cpu(), cal_residual, best["alpha"])
    calibration_result = calibrate_problem_folds(
        cal_rows, cal_risks, ratios, budgets=args.budgets, max_flip_rates=args.max_flip_rates,
        folds=args.calibration_folds, seed=args.seed,
    )
    metadata = {
        "schema": PHASE_B25B_SCHEMA,
        "seed": args.seed,
        "base_checkpoint": str(args.base_checkpoint),
        "base_schema": base_checkpoint["metadata"]["schema"],
        "base_variant": base_checkpoint["metadata"]["variant"],
        "base_checkpoint_selection_split": base_checkpoint["metadata"]["checkpoint_selection_split"],
        "pca_dim": transform["pca_dim"],
        "model_dim": args.model_dim,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "transform_fit_split": transform["fit_split"],
        "checkpoint_selection_split": "validation",
        "alpha_selection_split": "validation",
        "split_problem_counts": manifest["problem_counts"],
        "positive_weight": positive_weight,
        "best": best,
        "calibration": calibration_result,
        "ratios": ratios,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": metadata,
            "transform": transform,
            "frozen_base_checkpoint": base_checkpoint,
        },
        output,
    )
    write_json(args.metrics_output, metadata)


if __name__ == "__main__":
    main()
