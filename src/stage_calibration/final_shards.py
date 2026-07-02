from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


SHARD_SUMMARY_RE = re.compile(r"summary_shard_(\d+)_of_(\d+)\.json$")


def shard_tasks(tasks: list[dict[str, Any]], *, shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    validate_shard(shard_index=shard_index, shard_count=shard_count)
    output = []
    for index, task in enumerate(tasks):
        if index % shard_count != shard_index:
            continue
        row = dict(task)
        row["final_eval_index"] = index
        row["final_shard_index"] = shard_index
        row["final_shard_count"] = shard_count
        output.append(row)
    return output


def annotate_final_eval_indices(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, task in enumerate(tasks):
        row = dict(task)
        row["final_eval_index"] = index
        output.append(row)
    return output


def validate_shard(*, shard_index: int, shard_count: int) -> None:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must satisfy 0 <= index < count, got {shard_index}/{shard_count}")


def shard_summary_path(final_dir: str | Path, *, shard_index: int, shard_count: int) -> Path:
    validate_shard(shard_index=shard_index, shard_count=shard_count)
    return Path(final_dir) / f"summary_shard_{shard_index:05d}_of_{shard_count:05d}.json"


def shard_dataset_dir(
    final_dir: str | Path,
    dataset: str,
    *,
    shard_index: int,
    shard_count: int,
) -> Path:
    validate_shard(shard_index=shard_index, shard_count=shard_count)
    return (
        Path(final_dir)
        / dataset
        / "shards"
        / f"shard_{shard_index:05d}_of_{shard_count:05d}"
    )


def infer_shard_count(final_dir: str | Path) -> int:
    counts = set()
    for path in Path(final_dir).glob("summary_shard_*_of_*.json"):
        match = SHARD_SUMMARY_RE.match(path.name)
        if match:
            counts.add(int(match.group(2)))
    if not counts:
        raise FileNotFoundError(f"No shard summaries found in {final_dir}")
    if len(counts) != 1:
        raise ValueError(f"Conflicting shard counts found in {final_dir}: {sorted(counts)}")
    return counts.pop()


def completed_shard_indices(final_dir: str | Path, *, shard_count: int) -> set[int]:
    output = set()
    for path in Path(final_dir).glob(f"summary_shard_*_of_{shard_count:05d}.json"):
        match = SHARD_SUMMARY_RE.match(path.name)
        if match:
            output.add(int(match.group(1)))
    return output


def summarize_rows(rows: list[dict[str, Any]], *, method: dict[str, Any], seed: int) -> dict[str, Any]:
    correct = sum(int(row.get("correct", False)) for row in rows)
    stage_tokens = Counter()
    fallback = Counter()
    theoretical = []
    for row in rows:
        runtime = row.get("runtime_stage_mask", {})
        stage_tokens.update(runtime.get("tokens_by_stage", {}))
        if runtime.get("fallback_reason"):
            fallback[str(runtime["fallback_reason"])] += 1
        theoretical.append(float(runtime.get("theoretical_average_mlp_pruning_ratio", 0.0)))
    return {
        "method": method,
        "seed": int(seed),
        "problems": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "valid_stage_protocol_rate": (
            sum(int(row.get("stage_protocol", {}).get("valid", False)) for row in rows) / len(rows)
            if rows
            else None
        ),
        "fallback_rate": sum(fallback.values()) / len(rows) if rows else None,
        "fallback_reasons": dict(fallback),
        "truncation_rate": sum(int(row.get("truncated", False)) for row in rows) / len(rows) if rows else None,
        "mean_generated_tokens": (
            sum(float(row.get("generated_tokens", 0.0)) for row in rows) / len(rows)
            if rows
            else None
        ),
        "stage_tokens": dict(stage_tokens),
        "theoretical_average_mlp_pruning_ratio": (
            sum(theoretical) / len(theoretical) if theoretical else 0.0
        ),
    }


def merge_final_shards(
    *,
    final_dir: str | Path,
    shard_count: int,
    output_summary_path: str | Path,
    metadata: dict[str, Any],
    final_eval_limit: int | None,
    bootstrap_seed: int,
    policy_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_dir = Path(final_dir)
    missing = sorted(set(range(shard_count)) - completed_shard_indices(final_dir, shard_count=shard_count))
    if missing:
        raise FileNotFoundError(f"Missing final shard summaries: {missing}")

    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        shards_dir = dataset_dir / "shards"
        if not shards_dir.exists():
            continue
        groups: dict[tuple[str, int], dict[str, Any]] = {}
        for shard_index in range(shard_count):
            shard_dir = shard_dataset_dir(
                final_dir,
                dataset_dir.name,
                shard_index=shard_index,
                shard_count=shard_count,
            )
            if not shard_dir.exists():
                raise FileNotFoundError(f"Missing shard dataset directory: {shard_dir}")
            for summary_path in sorted(shard_dir.glob("*.summary.json")):
                summary = read_json(summary_path)
                method = summary["method"]
                seed = int(summary["seed"])
                key = (str(method["name"]), seed)
                rows_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
                entry = groups.setdefault(key, {"method": method, "seed": seed, "rows": []})
                entry["rows"].extend(read_jsonl(rows_path))

        datasets[dataset_dir.name] = []
        for (method_name, seed), entry in sorted(groups.items()):
            rows = sorted(
                entry["rows"],
                key=lambda row: (
                    int(row.get("final_eval_index", 10**12)),
                    str(row.get("id", "")),
                ),
            )
            combined_rows_path = dataset_dir / f"{method_name}_seed{seed}.jsonl"
            combined_summary_path = dataset_dir / f"{method_name}_seed{seed}.summary.json"
            summary = summarize_rows(rows, method=entry["method"], seed=seed)
            write_jsonl(combined_rows_path, rows)
            write_json(combined_summary_path, summary)
            datasets[dataset_dir.name].append(summary)

    aggregates = aggregate_final_summaries(final_dir=final_dir, datasets=datasets, bootstrap_seed=bootstrap_seed)
    final_summary = {
        "schema": "stage_calibrated_final_eval_v1",
        **metadata,
        "final_eval_limit": final_eval_limit,
        "final_sharded": True,
        "final_shard_count": shard_count,
        "policy_selection_used": policy_selection is not None,
        "policy_selection": (
            {
                "schema": policy_selection.get("schema"),
                "input_manifest_hash": policy_selection.get("input_manifest_hash"),
                "selected_policy_roles": sorted(policy_selection.get("selected_policies", {})),
            }
            if policy_selection is not None
            else None
        ),
        "evaluated_methods": sorted(
            {
                summary["method"]["name"]
                for summaries in datasets.values()
                for summary in summaries
            }
        ),
        "datasets": datasets,
        "aggregates": aggregates,
    }
    write_json(output_summary_path, final_summary)
    return final_summary


def aggregate_final_summaries(
    *,
    final_dir: str | Path,
    datasets: dict[str, list[dict[str, Any]]],
    bootstrap_seed: int,
    dataset_row_dirs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    final_dir = Path(final_dir)
    dataset_row_dirs = dataset_row_dirs or {}
    rng = random.Random(int(bootstrap_seed))
    aggregates = {}
    for dataset, summaries in datasets.items():
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for summary in summaries:
            by_method[summary["method"]["name"]].append(summary)
        aggregates[dataset] = {}
        for name, values in by_method.items():
            accuracies = [float(value["accuracy"]) for value in values if value["accuracy"] is not None]
            mean = sum(accuracies) / len(accuracies) if accuracies else None
            std = (
                math.sqrt(sum((value - mean) ** 2 for value in accuracies) / max(1, len(accuracies) - 1))
                if mean is not None
                else None
            )
            aggregates[dataset][name] = {
                "seeds": [value["seed"] for value in values],
                "accuracy_mean": mean,
                "accuracy_std": std,
                "theoretical_average_mlp_pruning_ratio_mean": (
                    sum(float(value["theoretical_average_mlp_pruning_ratio"]) for value in values)
                    / len(values)
                    if values
                    else 0.0
                ),
            }
        row_dir = Path(dataset_row_dirs.get(dataset, final_dir / dataset))
        dense_files = sorted(row_dir.glob("structured_dense_seed*.jsonl"))
        for name in by_method:
            if name == "structured_dense":
                continue
            method_files = sorted(row_dir.glob(f"{name}_seed*.jsonl"))
            paired_deltas = []
            for dense_file, method_file in zip(dense_files, method_files):
                dense_rows = read_jsonl(dense_file)
                method_rows = read_jsonl(method_file)
                paired_deltas.extend(
                    int(candidate["correct"]) - int(dense["correct"])
                    for dense, candidate in zip(dense_rows, method_rows)
                )
            if paired_deltas:
                bootstrap = []
                for _ in range(1000):
                    sample = [paired_deltas[rng.randrange(len(paired_deltas))] for _ in paired_deltas]
                    bootstrap.append(sum(sample) / len(sample))
                bootstrap.sort()
                aggregates[dataset][name]["paired_accuracy_delta_vs_structured_dense"] = (
                    sum(paired_deltas) / len(paired_deltas)
                )
                aggregates[dataset][name]["paired_bootstrap_95ci"] = [
                    bootstrap[int(0.025 * (len(bootstrap) - 1))],
                    bootstrap[int(0.975 * (len(bootstrap) - 1))],
                ]
    return aggregates


def ensure_clean_shard_dir(path: str | Path) -> Path:
    return ensure_dir(path)
