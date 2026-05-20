from __future__ import annotations

import argparse

from src.metrics.auc import safe_auc, spearman
from src.utils.io import read_jsonl, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    in_path = args.input or cfg["paths"]["counterfactuals"]
    out_path = args.output or cfg["paths"]["entropy_auc"]
    rows = read_jsonl(in_path)
    labels = [int(row["flipped"]) for row in rows]
    scores = [float(row["entropy"]) for row in rows]
    result = safe_auc(labels, scores)
    result["spearman"] = spearman(scores, labels)
    result["n"] = len(rows)
    result["positive_rate"] = sum(labels) / len(labels) if labels else None
    write_json(out_path, result)


if __name__ == "__main__":
    main()
