from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

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
    return sorted(out, key=lambda x: x[key])


def segment_layer_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["segment_id"]), int(row["layer_id"]))
        groups.setdefault(key, []).append(row)
    return [
        {"segment_id": segment_id, "layer_id": layer_id, "flip_rate": rate(group), "n": len(group)}
        for (segment_id, layer_id), group in sorted(groups.items())
    ]


def summarize_oracles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    layer_scores: dict[int, list[int]] = {}
    prompt_layer_scores: dict[tuple[str, int], list[int]] = {}
    step_layer_scores: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        flip = int(row["flipped"])
        layer_id = int(row["layer_id"])
        segment_id = int(row["segment_id"])
        layer_scores.setdefault(layer_id, []).append(flip)
        prompt_layer_scores.setdefault((row["id"], layer_id), []).append(flip)
        step_layer_scores.setdefault((row["id"], segment_id), []).append(flip)

    static_best = max((sum(v) / len(v), k) for k, v in layer_scores.items())
    prompt_best_by_id: dict[str, float] = {}
    for (item_id, _layer_id), values in prompt_layer_scores.items():
        prompt_best_by_id[item_id] = max(prompt_best_by_id.get(item_id, 0.0), sum(values) / len(values))
    step_best = [max(values) for values in step_layer_scores.values()]
    return {
        "static_oracle_best_layer": static_best[1],
        "static_oracle_flip_rate": static_best[0],
        "prompt_oracle_flip_rate": sum(prompt_best_by_id.values()) / len(prompt_best_by_id),
        "step_oracle_flip_rate": sum(step_best) / len(step_best),
        "n_counterfactuals": len(rows),
    }


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
    matrix = segment_layer_matrix(rows)
    summary = {
        "n": len(rows),
        "overall_flip_rate": rate(rows),
        "layer_flip_rates": layer_rates,
        "segment_flip_rates": segment_rates,
        "problem_flip_rates": problem_rates,
        "segment_layer_heatmap": matrix,
        "oracles": summarize_oracles(rows),
    }
    write_json(out_path, summary)
    write_csv(layer_csv, layer_rates)
    write_csv(matrix_csv, matrix)


if __name__ == "__main__":
    main()
