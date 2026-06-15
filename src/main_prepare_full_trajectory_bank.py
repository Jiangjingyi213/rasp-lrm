from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import ensure_dir, read_json, read_jsonl, read_yaml, write_json, write_jsonl


def source_isolated(config: dict[str, Any], source: str) -> bool:
    data = config["data"]
    if source == "gsm8k_train":
        return data.get("dataset") == "gsm8k" and data.get("split") == "train"
    if source == "math_train":
        return (
            data.get("dataset") == "math500"
            and data.get("dataset_label") == "math_train"
            and data.get("name_or_path") == "rasbt/math_full_minus_math500"
            and data.get("split") == "train"
        )
    return False


def boundary_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["id"]), int(row["generated_tokens_at_boundary"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-problems-per-source", type=int, required=True)
    parser.add_argument("--maximum-problems-per-source", type=int)
    args = parser.parse_args()
    if (
        args.maximum_problems_per_source is not None
        and args.maximum_problems_per_source < args.minimum_problems_per_source
    ):
        raise ValueError("maximum-problems-per-source cannot be below the required minimum")

    manifest = read_json(args.manifest)
    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(list)
    validations = []
    empty_shards = 0
    source_problems: dict[str, set[str]] = defaultdict(set)
    expected_ratios = None
    collection_signature = None
    for item in manifest:
        config = read_yaml(item["config"])
        if not source_isolated(config, str(item["source"])):
            raise ValueError(f"Training-source isolation failed: {item['config']}")
        ratios = [float(value) for value in config["aligned_window_bank"]["ratios"]]
        if expected_ratios is None:
            expected_ratios = ratios
        elif ratios != expected_ratios:
            raise ValueError("Full-trajectory shards use different action grids")
        signature = {
            name: config.get(name)
            for name in (
                "seed",
                "model",
                "prompt",
                "generation",
                "aligned_window_bank",
                "stage_sensitivity",
            )
        }
        if collection_signature is None:
            collection_signature = signature
        elif signature != collection_signature:
            raise ValueError("Full-trajectory shards use inconsistent collection settings")
        root = Path(item["run_dir"])
        validation = read_json(root / "07_full_trajectory_bank_validation.json")
        validations.append(validation)
        empty_shards += int(bool(validation.get("empty_shard")))
        if (
            validation.get("status") != "ok"
            or validation.get("boundary_sampling") != "causal_grid"
            or validation.get("action_terminal_semantics")
            != "eos_before_action_window_complete_v1"
            or not validation.get("stage_sensitivity_enabled")
            or validation.get("stage_sensitivity_diagnostic_only") is not True
            or validation.get("collection_config_fingerprint")
            != config_fingerprint(
                config,
                (
                    "seed",
                    "model",
                    "prompt",
                    "data",
                    "generation",
                    "aligned_window_bank",
                    "stage_sensitivity",
                ),
            )
        ):
            raise ValueError(f"Invalid full-trajectory shard: {root}")
        rows = read_jsonl(root / "05_full_trajectory_probe_dataset.jsonl")
        hidden = torch.load(root / "05_full_trajectory_hidden_states.pt", map_location="cpu")
        if len(rows) != len(hidden):
            raise ValueError(f"Row/hidden mismatch: {root}")
        for index, row in enumerate(rows):
            grouped[boundary_key(row)].append((row, hidden[index].detach().float().cpu()))
            source_problems[str(row["dataset"])].add(str(row["id"]))

    if expected_ratios is None:
        raise ValueError("Manifest contains no full-trajectory shards")
    output_rows: list[dict[str, Any]] = []
    output_hidden: list[torch.Tensor] = []
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_hidden: list[torch.Tensor] = []
    excluded_invalid_boundaries = 0
    for key, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: float(item[0]["ratio"]))
        rows = [item[0] for item in candidates]
        if [float(row["ratio"]) for row in rows] != expected_ratios:
            raise ValueError(f"{key}: incomplete action grid")
        reference = rows[0]
        complete = all(int(row["action_duration_tokens"]) == 16 for row in rows)
        valid_action_outcomes = all(
            bool(row.get("action_completed_or_terminal", False)) for row in rows
        )
        record = {
            "dataset": reference["dataset"],
            "id": reference["id"],
            "generated_tokens_at_boundary": int(reference["generated_tokens_at_boundary"]),
            "max_new_tokens": int(reference["max_new_tokens"]),
            "entropy": float(reference["entropy"]),
            "confidence": float(reference["confidence"]),
            "trusted_stage": str(reference.get("trusted_stage", "unknown")),
            "stage_confidence": float(reference.get("stage_confidence", 0.0)),
            "stage_probabilities": reference.get("stage_probabilities"),
            "candidate_ratios": expected_ratios,
            "candidate_flipped": [bool(row["flipped"]) for row in rows],
            "candidate_token_divergence": [
                float(row["window_token_divergence"]) for row in rows
            ],
            "candidate_action_duration_tokens": [
                int(row["action_duration_tokens"]) for row in rows
            ],
            "full_action_windows": complete,
            "valid_action_outcomes": valid_action_outcomes,
            "candidate_terminal_eos": [
                bool(row.get("action_terminal_eos", False)) for row in rows
            ],
            "boundary_role": str(reference.get("boundary_role", "unknown")),
            "controller_eligible": bool(reference.get("controller_eligible", False)),
            "causal_feature_schema": [
                "generated_tokens_at_boundary",
                "max_new_tokens",
                "entropy",
                "confidence",
                "stage_probabilities",
                "hidden_state",
                "candidate_ratio",
            ],
            "diagnostic_only": {
                "dense_trajectory_relative_position": float(reference.get("stage_position", 0.0)),
                "hard_operational_stage": str(reference.get("operational_stage", "unknown")),
                "boundary_role": str(reference.get("boundary_role", "unknown")),
            },
        }
        if not valid_action_outcomes and record["controller_eligible"]:
            excluded_invalid_boundaries += 1
            continue
        if record["controller_eligible"]:
            output_rows.append(record)
            output_hidden.append(candidates[0][1])
        else:
            diagnostic_rows.append(record)
            diagnostic_hidden.append(candidates[0][1])

    available_eligible_problems: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        available_eligible_problems[str(row["dataset"])].add(str(row["id"]))
    insufficient = {
        source: len(available_eligible_problems[source])
        for source in ("gsm8k", "math_train")
        if len(available_eligible_problems[source]) < args.minimum_problems_per_source
    }
    if insufficient:
        raise ValueError(
            f"Insufficient full-trajectory eligible problems: {insufficient}; "
            f"required={args.minimum_problems_per_source}"
        )
    selected_problems = {
        source: set(
            sorted(available_eligible_problems[source])[
                : args.maximum_problems_per_source
            ]
            if args.maximum_problems_per_source
            else sorted(available_eligible_problems[source])
        )
        for source in ("gsm8k", "math_train")
    }
    selected_keys = {
        (source, problem)
        for source, problems in selected_problems.items()
        for problem in problems
    }
    output_pairs = [
        (row, state)
        for row, state in zip(output_rows, output_hidden)
        if (str(row["dataset"]), str(row["id"])) in selected_keys
    ]
    output_rows = [row for row, _state in output_pairs]
    output_hidden = [state for _row, state in output_pairs]
    diagnostic_pairs = [
        (row, state)
        for row, state in zip(diagnostic_rows, diagnostic_hidden)
        if (str(row["dataset"]), str(row["id"])) in selected_keys
    ]
    diagnostic_rows = [row for row, _state in diagnostic_pairs]
    diagnostic_hidden = [state for _row, state in diagnostic_pairs]
    eligible_problems = selected_problems
    trusted_counts = Counter(str(row["trusted_stage"]) for row in output_rows)
    mean_soft_stage_probabilities = {
        stage: sum(
            float((row.get("stage_probabilities") or {}).get(stage) or 0.0)
            for row in output_rows
        )
        / max(1, len(output_rows))
        for stage in ("setup", "reasoning", "final")
    }
    ratio_flips = Counter()
    ratio_rows = Counter()
    ratio_terminal_eos = Counter()
    stage_ratio_flips = Counter()
    stage_ratio_rows = Counter()
    for row in output_rows:
        for ratio, flipped, terminal in zip(
            row["candidate_ratios"][1:],
            row["candidate_flipped"][1:],
            row["candidate_terminal_eos"][1:],
        ):
            name = f"{float(ratio):.2f}"
            ratio_rows[name] += 1
            ratio_flips[name] += int(flipped)
            ratio_terminal_eos[name] += int(terminal)
            stage_key = (str(row["trusted_stage"]), name)
            stage_ratio_rows[stage_key] += 1
            stage_ratio_flips[stage_key] += int(flipped)
    replay_mismatch_boundaries = sum(
        int(float(value.get("dense_replay_token_mismatch_rate", 0.0)) > 0.0)
        for value in validations
    )
    total_validated_boundaries = sum(
        int(value.get("controller_eligible_boundaries", 0)) for value in validations
    )
    estimated_replay_mismatches = sum(
        float(value.get("dense_replay_token_mismatch_rate", 0.0))
        * int(value.get("controller_eligible_boundaries", 0))
        for value in validations
    )
    output = ensure_dir(args.output_dir)
    write_jsonl(output / "01_full_trajectory_causal_dataset.jsonl", output_rows)
    torch.save(torch.stack(output_hidden), output / "01_full_trajectory_causal_hidden_states.pt")
    write_jsonl(output / "01_full_trajectory_diagnostic_dataset.jsonl", diagnostic_rows)
    if diagnostic_hidden:
        torch.save(
            torch.stack(diagnostic_hidden),
            output / "01_full_trajectory_diagnostic_hidden_states.pt",
        )
    write_json(
        output / "01_full_trajectory_data_summary.json",
        {
            "schema": "rasp_full_trajectory_dense_state_bank_v1",
            "problem_count": len({(row["dataset"], row["id"]) for row in output_rows}),
            "source_observed_dense_correct_problems": {
                source: len(values) for source, values in sorted(source_problems.items())
            },
            "source_eligible_problems": {
                source: len(values) for source, values in sorted(eligible_problems.items())
            },
            "source_available_eligible_problems": {
                source: len(values)
                for source, values in sorted(available_eligible_problems.items())
            },
            "minimum_problems_per_source": args.minimum_problems_per_source,
            "maximum_problems_per_source": args.maximum_problems_per_source,
            "causal_boundary_count": len(output_rows),
            "tail_diagnostic_boundary_count": len(diagnostic_rows),
            "excluded_invalid_boundaries": excluded_invalid_boundaries,
            "trusted_stage_counts": dict(sorted(trusted_counts.items())),
            "mean_soft_stage_probabilities": mean_soft_stage_probabilities,
            "dose_response": [
                {
                    "ratio": float(name),
                    "rows": ratio_rows[name],
                    "flips": ratio_flips[name],
                    "flip_rate": ratio_flips[name] / ratio_rows[name],
                    "terminal_eos": ratio_terminal_eos[name],
                    "terminal_eos_rate": ratio_terminal_eos[name] / ratio_rows[name],
                }
                for name in sorted(ratio_rows)
            ],
            "trusted_stage_ratio_response": [
                {
                    "trusted_stage": stage,
                    "ratio": float(name),
                    "rows": stage_ratio_rows[(stage, name)],
                    "flips": stage_ratio_flips[(stage, name)],
                    "flip_rate": stage_ratio_flips[(stage, name)]
                    / stage_ratio_rows[(stage, name)],
                }
                for stage, name in sorted(stage_ratio_rows)
            ],
            "shards": len(validations),
            "empty_shards": empty_shards,
            "shards_with_dense_replay_mismatch": replay_mismatch_boundaries,
            "dense_replay_mismatch_rate": (
                estimated_replay_mismatches / total_validated_boundaries
                if total_validated_boundaries
                else 0.0
            ),
            "final_test_sources_used": False,
            "causal_features_exclude_dense_trajectory_length": True,
            "tail_anchor_is_diagnostic_only": True,
            "causal_soft_stage_verified": all(
                bool(value.get("stage_probe_causal")) for value in validations
            ),
            "dense_correctness_recomputed_from_completion": True,
        },
    )


if __name__ == "__main__":
    main()
