from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_json, read_yaml, write_json


def collect_one(config_path: str) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    paths = cfg["paths"]
    summary = read_json(paths["heatmap_summary"])
    entropy = read_json(paths["entropy_auc"])
    probe = read_json(paths["probe_metrics"])
    return {
        "config": config_path,
        "dataset": cfg["data"]["dataset"],
        "run_dir": paths["run_dir"],
        "n": summary["n"],
        "overall_flip_rate": summary["overall_flip_rate"],
        "oracles": summary["oracles"],
        "entropy_auc": entropy,
        "probe_metrics": probe,
    }


def weighted_mean(items: list[dict[str, Any]], key: str) -> float | None:
    total_n = sum(int(item["n"]) for item in items)
    if total_n == 0:
        return None
    return sum(float(item[key]) * int(item["n"]) for item in items) / total_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = [collect_one(path) for path in args.configs]
    combined = {
        "runs": results,
        "micro_overall_flip_rate": weighted_mean(results, "overall_flip_rate"),
        "total_counterfactuals": sum(int(item["n"]) for item in results),
    }
    write_json(Path(args.output), combined)


if __name__ == "__main__":
    main()
