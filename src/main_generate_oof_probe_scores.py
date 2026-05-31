from __future__ import annotations

import argparse
from pathlib import Path

from src.probes.oof_scores import generate_oof_scores
from src.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--feature-set",
        default="hidden",
        choices=["hidden", "activation", "entropy", "confidence", "combined", "action_hidden", "action_hidden_stage"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows, summary = generate_oof_scores(
        run_dirs=args.run_dirs,
        feature_set=args.feature_set,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)
    summary_output = args.summary_output or str(Path(args.output).with_suffix(".summary.json"))
    write_json(summary_output, summary)


if __name__ == "__main__":
    main()
