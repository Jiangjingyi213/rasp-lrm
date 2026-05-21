from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.main_heatmap_summary import grouped_rates, grouped_rates_multi, rate, segment_layer_matrix
from src.metrics.oracles import summarize_oracles
from src.utils.io import read_json, read_jsonl, read_yaml, write_json


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
        "heatmap_summary": summary,
        "oracles": summary["oracles"],
        "entropy_auc": entropy,
        "probe_metrics": probe,
    }


def collect_rows(config_path: str) -> list[dict[str, Any]]:
    cfg = read_yaml(config_path)
    dataset = cfg["data"]["dataset"]
    rows = []
    for row in read_jsonl(cfg["paths"]["counterfactuals"]):
        row = dict(row)
        row["dataset"] = row.get("dataset") or dataset
        row["_combined_id"] = f"{row['dataset']}:{row['id']}"
        rows.append(row)
    return rows


def weighted_mean(items: list[dict[str, Any]], key: str) -> float | None:
    total_n = sum(int(item["n"]) for item in items)
    if total_n == 0:
        return None
    return sum(float(item[key]) * int(item["n"]) for item in items) / total_n


def combined_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_rows = []
    for row in rows:
        oracle_row = dict(row)
        oracle_row["id"] = row["_combined_id"]
        oracle_rows.append(oracle_row)
    return {
        "n": len(rows),
        "overall_flip_rate": rate(rows),
        "dataset_flip_rates": grouped_rates(rows, "dataset"),
        "layer_flip_rates": grouped_rates(rows, "layer_id"),
        "segment_flip_rates": grouped_rates(rows, "segment_id"),
        "problem_flip_rates": grouped_rates(rows, "_combined_id"),
        "module_flip_rates": grouped_rates(rows, "module"),
        "ratio_flip_rates": grouped_rates(rows, "ratio"),
        "segment_type_flip_rates": grouped_rates(rows, "segment_type"),
        "dataset_module_flip_rates": grouped_rates_multi(rows, ["dataset", "module"]),
        "dataset_ratio_flip_rates": grouped_rates_multi(rows, ["dataset", "ratio"]),
        "module_ratio_flip_rates": grouped_rates_multi(rows, ["module", "ratio"]),
        "segment_type_module_ratio_flip_rates": grouped_rates_multi(rows, ["segment_type", "module", "ratio"]),
        "dataset_segment_type_module_ratio_flip_rates": grouped_rates_multi(
            rows, ["dataset", "segment_type", "module", "ratio"]
        ),
        "segment_layer_heatmap": segment_layer_matrix(rows),
        "oracles": summarize_oracles(oracle_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = [collect_one(path) for path in args.configs]
    rows = [row for path in args.configs for row in collect_rows(path)]
    combined = {
        "runs": results,
        "combined_summary": combined_summary(rows),
        "micro_overall_flip_rate": weighted_mean(results, "overall_flip_rate"),
        "total_counterfactuals": sum(int(item["n"]) for item in results),
    }
    write_json(Path(args.output), combined)


if __name__ == "__main__":
    main()
