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
    dense_observation_tokens = Counter()
    masked_tokens = Counter()
    mask_refresh_counts = Counter()
    fallback = Counter()
    theoretical = []
    actual = []
    actual_pruning_accounting = None
    runtime_backend = None
    runtime_alpha = None
    runtime_warmup_tokens = None
    runtime_score_mode = None
    runtime_runtime_weight = None
    runtime_prior_weight = None
    runtime_protected_core_ratios = None
    runtime_refresh_intervals = None
    runtime_window_tokens = None
    runtime_base_policy = None
    runtime_stage_prior_policy = None
    runtime_static_core_ratios = None
    runtime_swap_ratios = None
    runtime_actual_swapped_channels = None
    runtime_baseline_type = None
    runtime_selection_method = None
    runtime_prune_ratio = None
    runtime_density = None
    prompt_dense_tokens = 0
    decode_masked_tokens = 0
    keep_ratios_by_layer = None
    pruning_granularity = None
    wanda_weight_sparsity = None
    wanda_sparsity_ratio = None
    wanda_calibration_samples = None
    wanda_calibration_source = None
    wanda_target_modules = None
    weight_sparsity_by_module = None
    matched_rasp_reference = None
    target_matched_to_rasp_actual_mlp_pruning = None
    for row in rows:
        runtime = row.get("runtime_stage_mask", {})
        runtime_backend = runtime_backend or runtime.get("backend")
        runtime_baseline_type = runtime_baseline_type or runtime.get("baseline_type")
        runtime_selection_method = runtime_selection_method or runtime.get("selection_method")
        runtime_prune_ratio = (
            runtime_prune_ratio if runtime_prune_ratio is not None else runtime.get("prune_ratio")
        )
        runtime_density = runtime_density if runtime_density is not None else runtime.get("density")
        prompt_dense_tokens += int(runtime.get("prompt_dense_tokens", 0))
        decode_masked_tokens += int(runtime.get("decode_masked_tokens", 0))
        keep_ratios_by_layer = keep_ratios_by_layer or runtime.get("keep_ratios_by_layer")
        pruning_granularity = pruning_granularity or runtime.get("pruning_granularity")
        wanda_weight_sparsity = (
            wanda_weight_sparsity
            if wanda_weight_sparsity is not None
            else runtime.get("wanda_weight_sparsity")
        )
        wanda_sparsity_ratio = (
            wanda_sparsity_ratio
            if wanda_sparsity_ratio is not None
            else runtime.get("wanda_sparsity_ratio")
        )
        wanda_calibration_samples = (
            wanda_calibration_samples
            if wanda_calibration_samples is not None
            else runtime.get("wanda_calibration_samples")
        )
        wanda_calibration_source = wanda_calibration_source or runtime.get("wanda_calibration_source")
        wanda_target_modules = wanda_target_modules or runtime.get("wanda_target_modules")
        weight_sparsity_by_module = weight_sparsity_by_module or runtime.get("weight_sparsity_by_module")
        matched_rasp_reference = matched_rasp_reference or runtime.get("matched_rasp_reference")
        target_matched_to_rasp_actual_mlp_pruning = (
            target_matched_to_rasp_actual_mlp_pruning
            if target_matched_to_rasp_actual_mlp_pruning is not None
            else runtime.get("target_matched_to_rasp_actual_mlp_pruning")
        )
        runtime_alpha = runtime_alpha if runtime_alpha is not None else runtime.get("alpha")
        runtime_warmup_tokens = runtime_warmup_tokens or runtime.get("warmup_tokens")
        runtime_score_mode = runtime_score_mode or runtime.get("score_mode")
        runtime_runtime_weight = (
            runtime_runtime_weight
            if runtime_runtime_weight is not None
            else runtime.get("runtime_weight")
        )
        runtime_prior_weight = (
            runtime_prior_weight
            if runtime_prior_weight is not None
            else runtime.get("prior_weight")
        )
        runtime_protected_core_ratios = (
            runtime_protected_core_ratios or runtime.get("protected_core_ratios")
        )
        runtime_refresh_intervals = runtime_refresh_intervals or runtime.get("refresh_intervals")
        runtime_window_tokens = runtime_window_tokens or runtime.get("window_tokens")
        runtime_base_policy = runtime_base_policy or runtime.get("base_policy")
        runtime_stage_prior_policy = runtime_stage_prior_policy or runtime.get("stage_prior_policy")
        runtime_static_core_ratios = runtime_static_core_ratios or runtime.get("static_core_ratios")
        runtime_swap_ratios = runtime_swap_ratios or runtime.get("swap_ratios")
        runtime_actual_swapped_channels = (
            runtime_actual_swapped_channels
            or runtime.get("actual_swapped_channels_by_stage_layer")
        )
        stage_tokens.update(runtime.get("tokens_by_stage", {}))
        dense_observation_tokens.update(runtime.get("dense_observation_tokens_by_stage", {}))
        masked_tokens.update(runtime.get("masked_tokens_by_stage", {}))
        mask_refresh_counts.update(runtime.get("mask_refresh_count_by_stage", {}))
        if runtime.get("fallback_reason"):
            fallback[str(runtime["fallback_reason"])] += 1
        theoretical.append(float(runtime.get("theoretical_average_mlp_pruning_ratio", 0.0)))
        actual.append(
            float(
                runtime.get(
                    "actual_average_mlp_pruning_ratio",
                    runtime.get("theoretical_average_mlp_pruning_ratio", 0.0),
                )
            )
        )
        actual_pruning_accounting = (
            actual_pruning_accounting
            or runtime.get("actual_pruning_accounting")
            or "estimated_from_stage_ratios"
        )
    summary = {
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
        "actual_average_mlp_pruning_ratio": sum(actual) / len(actual) if actual else 0.0,
        "actual_pruning_accounting": actual_pruning_accounting or "estimated_from_stage_ratios",
    }
    if "target_pruning_ratio" in method:
        target = float(method["target_pruning_ratio"])
        actual_value = float(summary["actual_average_mlp_pruning_ratio"])
        summary["target_pruning_ratio"] = target
        summary["target_pruning_reached"] = bool(actual_value >= target)
        summary["target_pruning_gap"] = actual_value - target
        summary["target_pruning_status"] = (
            "passed" if actual_value >= target else "target_pruning_not_reached"
        )
    if "target_pruning_label" in method:
        summary["target_pruning_label"] = method["target_pruning_label"]
    if runtime_backend:
        summary["runtime_backend"] = runtime_backend
    if runtime_baseline_type:
        summary["runtime_baseline_type"] = runtime_baseline_type
    if runtime_selection_method:
        summary["runtime_selection_method"] = runtime_selection_method
    if runtime_prune_ratio is not None:
        summary["runtime_prune_ratio"] = runtime_prune_ratio
    if runtime_density is not None:
        summary["runtime_density"] = runtime_density
    if prompt_dense_tokens:
        summary["prompt_dense_tokens"] = prompt_dense_tokens
    if decode_masked_tokens:
        summary["decode_masked_tokens"] = decode_masked_tokens
    if keep_ratios_by_layer:
        summary["keep_ratios_by_layer"] = keep_ratios_by_layer
    if pruning_granularity:
        summary["pruning_granularity"] = pruning_granularity
    if wanda_weight_sparsity is not None:
        summary["wanda_weight_sparsity"] = wanda_weight_sparsity
    if wanda_sparsity_ratio is not None:
        summary["wanda_sparsity_ratio"] = wanda_sparsity_ratio
    if wanda_calibration_samples is not None:
        summary["wanda_calibration_samples"] = wanda_calibration_samples
    if wanda_calibration_source:
        summary["wanda_calibration_source"] = wanda_calibration_source
    if wanda_target_modules:
        summary["wanda_target_modules"] = wanda_target_modules
    if weight_sparsity_by_module:
        summary["weight_sparsity_by_module"] = weight_sparsity_by_module
    if matched_rasp_reference:
        summary["matched_rasp_reference"] = matched_rasp_reference
    if target_matched_to_rasp_actual_mlp_pruning is not None:
        summary["target_matched_to_rasp_actual_mlp_pruning"] = (
            target_matched_to_rasp_actual_mlp_pruning
        )
    if dense_observation_tokens:
        summary["dense_observation_tokens_by_stage"] = dict(dense_observation_tokens)
    if masked_tokens:
        summary["masked_tokens_by_stage"] = dict(masked_tokens)
    if mask_refresh_counts:
        summary["mask_refresh_count_by_stage"] = dict(mask_refresh_counts)
    if runtime_alpha is not None:
        summary["adaptive_alpha"] = runtime_alpha
    if runtime_warmup_tokens is not None:
        summary["adaptive_warmup_tokens"] = runtime_warmup_tokens
    if runtime_score_mode is not None:
        summary["adaptive_score_mode"] = runtime_score_mode
    if runtime_runtime_weight is not None:
        summary["adaptive_runtime_weight"] = runtime_runtime_weight
    if runtime_prior_weight is not None:
        summary["adaptive_prior_weight"] = runtime_prior_weight
    if runtime_protected_core_ratios is not None:
        summary["adaptive_protected_core_ratios"] = runtime_protected_core_ratios
    if runtime_refresh_intervals is not None:
        summary["adaptive_refresh_intervals"] = runtime_refresh_intervals
    if runtime_window_tokens is not None:
        summary["adaptive_window_tokens"] = runtime_window_tokens
    if runtime_base_policy is not None:
        summary["adaptive_base_policy"] = runtime_base_policy
    if runtime_stage_prior_policy is not None:
        summary["adaptive_stage_prior_policy"] = runtime_stage_prior_policy
    if runtime_static_core_ratios is not None:
        summary["adaptive_static_core_ratios"] = runtime_static_core_ratios
    if runtime_swap_ratios is not None:
        summary["adaptive_swap_ratios"] = runtime_swap_ratios
    if runtime_actual_swapped_channels is not None:
        summary["adaptive_actual_swapped_channels_by_stage_layer"] = runtime_actual_swapped_channels
    return summary


def _target_pruning_summary(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, summaries in sorted(datasets.items()):
        static_by_target = {
            str(summary.get("target_pruning_label")): summary
            for summary in summaries
            if str(summary["method"]["name"]).startswith("static_")
            and summary.get("target_pruning_label") is not None
        }
        for summary in sorted(summaries, key=lambda row: str(row["method"]["name"])):
            target = summary.get("target_pruning_label")
            if target is None:
                continue
            static = static_by_target.get(str(target))
            accuracy = summary.get("accuracy")
            static_accuracy = static.get("accuracy") if static else None
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "method": summary["method"]["name"],
                    "accuracy": accuracy,
                    "actual_average_mlp_pruning_ratio": summary.get(
                        "actual_average_mlp_pruning_ratio"
                    ),
                    "theoretical_average_mlp_pruning_ratio": summary.get(
                        "theoretical_average_mlp_pruning_ratio"
                    ),
                    "delta_vs_static_same_target": (
                        float(accuracy) - float(static_accuracy)
                        if accuracy is not None and static_accuracy is not None
                        else None
                    ),
                    "fallback_rate": summary.get("fallback_rate"),
                    "truncation_rate": summary.get("truncation_rate"),
                    "mean_generated_tokens": summary.get("mean_generated_tokens"),
                    "target_pruning_status": summary.get("target_pruning_status"),
                }
            )
    return rows


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
        "target_pruning_summary": _target_pruning_summary(datasets),
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
                "actual_average_mlp_pruning_ratio_mean": (
                    sum(
                        float(
                            value.get(
                                "actual_average_mlp_pruning_ratio",
                                value["theoretical_average_mlp_pruning_ratio"],
                            )
                        )
                        for value in values
                    )
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
