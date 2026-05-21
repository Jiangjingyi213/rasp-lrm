from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.metrics.oracles import summarize_oracles
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_json


def rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(int(row["flipped"]) for row in rows) / len(rows)


def grouped_rates(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    out = []
    for value, group in groups.items():
        out.append({key: value, "flip_rate": rate(group), "n": len(group)})
    return sorted(out, key=lambda x: str(x[key]))


def segment_layer_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        layer_key = ",".join(str(x) for x in row.get("pruned_layers") or [row.get("layer_id")])
        key = (int(row["segment_id"]), layer_key)
        groups.setdefault(key, []).append(row)
    return [
        {"segment_id": segment_id, "layer_key": layer_key, "flip_rate": rate(group), "n": len(group)}
        for (segment_id, layer_key), group in sorted(groups.items())
    ]


def grouped_rates_multi(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out = []
    for values, group in groups.items():
        record = {key: value for key, value in zip(keys, values)}
        record["flip_rate"] = rate(group)
        record["n"] = len(group)
        out.append(record)
    return sorted(out, key=lambda x: tuple(str(x[key]) for key in keys))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    paths = cfg["paths"]
    in_path = args.input or paths["counterfactuals"]
    out_path = args.output or paths.get("heatmap_summary", str(Path(in_path).with_name("06_heatmap_summary.json")))
    layer_csv = paths.get("layer_flip_rates", str(Path(in_path).with_name("06_layer_flip_rates.csv")))
    matrix_csv = paths.get("segment_layer_heatmap", str(Path(in_path).with_name("06_segment_layer_heatmap.csv")))

    rows = read_jsonl(in_path)
    layer_rates = grouped_rates(rows, "layer_id")
    segment_rates = grouped_rates(rows, "segment_id")
    problem_rates = grouped_rates(rows, "id")
    module_rates = grouped_rates(rows, "module")
    ratio_rates = grouped_rates(rows, "ratio")
    segment_type_rates = grouped_rates(rows, "segment_type")
    module_ratio_rates = grouped_rates_multi(rows, ["module", "ratio"])
    segment_type_module_ratio_rates = grouped_rates_multi(rows, ["segment_type", "module", "ratio"])
    matrix = segment_layer_matrix(rows)
    summary = {
        "n": len(rows),
        "overall_flip_rate": rate(rows),
        "layer_flip_rates": layer_rates,
        "segment_flip_rates": segment_rates,
        "problem_flip_rates": problem_rates,
        "module_flip_rates": module_rates,
        "ratio_flip_rates": ratio_rates,
        "segment_type_flip_rates": segment_type_rates,
        "module_ratio_flip_rates": module_ratio_rates,
        "segment_type_module_ratio_flip_rates": segment_type_module_ratio_rates,
        "segment_layer_heatmap": matrix,
        "oracles": summarize_oracles(rows),
    }
    write_json(out_path, summary)
    write_csv(layer_csv, layer_rates)
    write_csv(matrix_csv, matrix)


if __name__ == "__main__":
    main()
