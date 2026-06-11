from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rasp.stage_probe import (
    STAGES,
    STAGE_PROBE_SCHEMA,
    STAGE_VARIANTS,
    StageProbeNet,
    fit_stage_transform,
    indices_for_stage_split,
    stage_index,
    stage_metrics,
    transform_stage_features,
    validate_stage_manifest,
)
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: StageProbeNet, features: torch.Tensor, labels: torch.Tensor, indices: list[int], device: torch.device):
    model.eval()
    logits = model(features[indices].to(device))
    predictions = logits.argmax(dim=1).cpu().tolist()
    values = labels[indices].tolist()
    loss = torch.nn.functional.cross_entropy(logits, labels[indices].to(device))
    return {"loss": float(loss.item()), **stage_metrics(values, predictions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=sorted(STAGE_VARIANTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(args.dataset)
    hidden = torch.load(args.hidden_states, map_location="cpu")
    manifest = read_json(args.manifest)
    validate_stage_manifest(rows, manifest, args.seed)
    train_indices = indices_for_stage_split(rows, manifest, "train")
    validation_indices = indices_for_stage_split(rows, manifest, "validation")
    transform = fit_stage_transform(rows, hidden, train_indices, pca_dim=args.pca_dim)
    features = transform_stage_features(rows, hidden, transform, args.variant)
    labels = torch.tensor([stage_index(str(row["stage"])) for row in rows], dtype=torch.long)
    counts = torch.bincount(labels[train_indices], minlength=len(STAGES)).float()
    class_weights = counts.sum() / (len(STAGES) * counts.clamp_min(1.0))
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = STAGE_VARIANTS[args.variant]
    model = StageProbeNet(features.shape[1], model_type, args.model_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best = {"macro_f1": -1.0, "loss": float("inf")}
    output = Path(args.output)
    ensure_dir(output.parent)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, features, labels, validation_indices, device)
        if (
            metrics["macro_f1"] > best["macro_f1"] + 1e-12
            or metrics["macro_f1"] == best["macro_f1"] and metrics["loss"] < best["loss"]
        ):
            best = {**metrics, "epoch": epoch}
            torch.save({"model": model.state_dict()}, output)
    model.load_state_dict(torch.load(output, map_location=device)["model"])
    metadata = {
        "schema": STAGE_PROBE_SCHEMA,
        "variant": args.variant,
        "model_type": model_type,
        "dim": int(features.shape[1]),
        "model_dim": args.model_dim,
        "pca_dim": transform["pca_dim"],
        "seed": args.seed,
        "manifest": str(args.manifest),
        "checkpoint_selection_split": "validation",
        "class_weights": class_weights.tolist(),
        "split_problem_counts": manifest["problem_counts"],
        "best_validation": best,
    }
    torch.save({"model": model.state_dict(), "transform": transform, "metadata": metadata}, output)
    write_json(args.metrics_output, metadata)


if __name__ == "__main__":
    main()
