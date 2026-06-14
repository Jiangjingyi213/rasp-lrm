from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.answer_match import answer_match
from src.rasp.window_sampling import boundary_positions
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_jsonl, read_yaml, write_json


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row.get("dataset") or "unknown"), str(row["id"]), int(row["boundary_index"])


def validate_aligned_window_bank(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    cfg = config.get("aligned_window_bank", {})
    rows = read_jsonl(paths["counterfactuals"])
    probe_rows = read_jsonl(paths["probe_dataset"])
    ratios = [float(value) for value in cfg.get("ratios", [])]
    expected = {f"{value:.8f}" for value in ratios}
    errors = []
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 512))
    if len(rows) != len(probe_rows):
        errors.append("Counterfactual and probe row counts differ")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
        if row.get("action_scope") != "single_fixed_window_then_dense":
            errors.append("Aligned bank contains a non-window action scope")
            break
        if row.get("action_window_alignment") != "affected_next_token_decisions_v2":
            errors.append("Aligned bank contains legacy or unknown action-window alignment")
            break
        expected_position = int(row.get("generated_tokens_at_boundary", -1)) / max(1, max_new_tokens)
        if int(row.get("max_new_tokens", -1)) != max_new_tokens or abs(float(row.get("position", -1.0)) - expected_position) > 1e-9:
            errors.append("Aligned bank position does not match runtime generated_tokens/max_new_tokens")
            break
        if row.get("ranking_scope") != "initial_prompt_prefill_fixed":
            errors.append("Aligned bank contains a non-runtime ranking scope")
            break
        if "window_token_divergence" not in row or "window_end_hidden_l2" not in row:
            errors.append("Aligned bank is missing local drift labels")
            break
    incomplete = sum(
        int({f"{float(row['ratio']):.8f}" for row in values} != expected or len(values) != len(ratios))
        for values in grouped.values()
    )
    if incomplete:
        errors.append(f"{incomplete} boundaries do not contain the complete ratio grid")
    trajectory_path = paths.get("trajectories")
    if trajectory_path and Path(trajectory_path).exists():
        window_tokens = int(cfg.get("window_tokens", 16))
        max_boundaries = cfg.get("max_boundaries_per_example")
        boundary_sampling = str(cfg.get("boundary_sampling", "prefix"))
        explicit_boundary_positions = cfg.get("boundary_positions")
        decision_start = cfg.get("decision_start")
        decision_stride = cfg.get("decision_stride")
        include_tail_anchor = bool(cfg.get("include_tail_anchor", False))
        expected_boundary_keys = set()
        for trajectory in read_jsonl(trajectory_path):
            if not answer_match(
                str(trajectory.get("completion", "")),
                str(trajectory.get("gold", "")),
            ):
                continue
            token_ids = trajectory.get("generated_token_ids")
            if token_ids is None:
                errors.append("Validated aligned bank trajectory is missing generated_token_ids")
                break
            positions = [
                position
                for position in boundary_positions(
                    len(token_ids),
                    window_tokens,
                    max_boundaries,
                    boundary_sampling,
                    explicit_boundary_positions,
                    decision_start=decision_start,
                    decision_stride=decision_stride,
                    include_tail_anchor=include_tail_anchor,
                )
                if position < max_new_tokens
            ]
            expected_boundary_keys.update(
                (str(trajectory.get("dataset") or "unknown"), str(trajectory["id"]), index)
                for index, _position in enumerate(positions)
            )
        if set(grouped) != expected_boundary_keys:
            errors.append(
                "Aligned bank boundary coverage does not match configured dense-correct trajectories"
            )
    hidden_indices = [int(row["hidden_index"]) for row in probe_rows]
    if hidden_indices != list(range(len(probe_rows))):
        errors.append("Probe hidden indices are not contiguous")
    if not Path(paths["probe_hidden_states"]).exists():
        errors.append("Probe hidden-state tensor is missing")
    dense_rows = [row for row in rows if abs(float(row["ratio"])) < 1e-12]
    dense_flip_rate = sum(int(bool(row["flipped"])) for row in dense_rows) / max(1, len(dense_rows))
    if dense_flip_rate > 0.0:
        errors.append(f"Dense controls must have zero paired flip rate, got {dense_flip_rate:.4f}")
    replay_flip_rate = sum(
        int(bool(row.get("dense_control_flipped_from_baseline"))) for row in dense_rows
    ) / max(1, len(dense_rows))
    controller_dense_rows = [
        row for row in dense_rows if bool(row.get("controller_eligible", True))
    ]
    dense_replay_token_mismatch_rate = sum(
        int(
            not bool(row.get("dense_replay_token_window_exact", True))
            or int(row.get("dense_replay_prefix_mismatches", 0)) > 0
        )
        for row in controller_dense_rows
    ) / max(1, len(controller_dense_rows))
    max_replay_flip_rate = float(cfg.get("max_dense_replay_flip_rate", 0.05))
    if replay_flip_rate > max_replay_flip_rate:
        errors.append(
            f"Dense replay flip rate is high: {replay_flip_rate:.4f} > {max_replay_flip_rate:.4f}"
        )
    if (
        str(cfg.get("boundary_sampling", "prefix")) == "causal_grid"
        and dense_replay_token_mismatch_rate > max_replay_flip_rate
    ):
        errors.append(
            "Dense replay token mismatch rate is high: "
            f"{dense_replay_token_mismatch_rate:.4f} > {max_replay_flip_rate:.4f}"
        )
    ratio_counts = Counter(f"{float(row['ratio']):.2f}" for row in rows)
    token_sources = sorted({str(row.get("boundary_token_source")) for row in rows})
    if token_sources != ["trajectory_generated_token_ids"]:
        errors.append(f"Formal aligned bank must use original generated token IDs, got {token_sources}")
    stage_cfg = config.get("stage_sensitivity")
    stage_counts = Counter()
    trusted_stage_counts = Counter()
    boundary_role_counts = Counter(str(row.get("boundary_role", "legacy")) for row in dense_rows)
    controller_eligible_rows = controller_dense_rows
    incomplete_action_rows = sum(
        int(int(row.get("action_duration_tokens", 0)) < int(cfg.get("window_tokens", 16)))
        for row in rows
    )
    invalid_controller_action_rows = sum(
        int(
            bool(row.get("controller_eligible", True))
            and not bool(row.get("action_completed_or_terminal", False))
        )
        for row in rows
    )
    if str(cfg.get("boundary_sampling", "prefix")) == "causal_grid":
        if any(int(row["generated_tokens_at_boundary"]) == 0 for row in controller_eligible_rows):
            errors.append("causal_grid must not include token 0")
        if any(bool(row.get("diagnostic_only")) for row in controller_eligible_rows):
            errors.append("Controller-eligible causal_grid row cannot be diagnostic-only")
        if invalid_controller_action_rows:
            errors.append(
                f"{invalid_controller_action_rows} controller-eligible causal-grid actions "
                "are neither complete nor terminal"
            )
        if any(
            bool(row.get("controller_eligible", True))
            and not bool(row.get("dense_restored_after_window"))
            and not bool(row.get("action_terminal_eos"))
            for row in rows
        ):
            errors.append("A causal-grid action did not restore dense after one window")
    if stage_cfg:
        valid_stages = {"setup", "reasoning", "verification", "final"}
        for row in dense_rows:
            stage = row.get("operational_stage")
            stage_counts[str(stage)] += 1
            trusted_stage_counts[str(row.get("trusted_stage", "missing"))] += 1
            if stage not in valid_stages:
                errors.append(f"Stage sensitivity row has invalid operational stage: {stage}")
                break
            if "stage_source" not in row or "reasoning_accepted" not in row:
                errors.append("Stage sensitivity row is missing stage decision metadata")
                break
            if bool(stage_cfg.get("require_causal_features", False)) and not bool(
                row.get("stage_probe_causal", False)
            ):
                errors.append("Stage sensitivity row uses a non-causal stage checkpoint")
                break
            if (
                row.get("stage_position_definition")
                != "generated_tokens_over_dense_trajectory_tokens_minus_one"
                or not 0.0 <= float(row.get("stage_position", -1.0)) <= 1.0
            ):
                errors.append("Stage sensitivity row uses an invalid or legacy stage position")
                break
            if stage != "verification":
                probabilities = row.get("stage_probabilities")
                if not isinstance(probabilities, dict) or set(probabilities) != {"setup", "reasoning", "final"}:
                    errors.append("Learned stage row is missing the three-class probability vector")
                    break
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "boundaries": len(grouped),
        "counterfactual_rows": len(rows),
        "ratios": ratios,
        "ratio_counts": dict(sorted(ratio_counts.items())),
        "dense_control_paired_flip_rate": dense_flip_rate,
        "dense_replay_flip_rate_from_baseline": replay_flip_rate,
        "dense_replay_token_mismatch_rate": dense_replay_token_mismatch_rate,
        "configured_window_tokens": int(cfg.get("window_tokens", 16)),
        "configured_max_new_tokens": max_new_tokens,
        "configured_max_boundaries_per_example": cfg.get("max_boundaries_per_example"),
        "boundary_sampling": str(cfg.get("boundary_sampling", "prefix")),
        "configured_boundary_positions": cfg.get("boundary_positions"),
        "configured_decision_start": cfg.get("decision_start"),
        "configured_decision_stride": cfg.get("decision_stride"),
        "configured_include_tail_anchor": bool(cfg.get("include_tail_anchor", False)),
        "boundary_role_counts": dict(sorted(boundary_role_counts.items())),
        "controller_eligible_boundaries": len(controller_eligible_rows),
        "incomplete_action_rows": incomplete_action_rows,
        "invalid_controller_action_rows": invalid_controller_action_rows,
        "action_scope": "single_fixed_window_then_dense",
        "action_window_alignment": "affected_next_token_decisions_v2",
        "ranking_scope": "initial_prompt_prefill_fixed",
        "boundary_token_sources": token_sources,
        "stage_sensitivity_enabled": bool(stage_cfg),
        "stage_sensitivity_diagnostic_only": (
            bool(stage_cfg.get("diagnostic_only", False)) if stage_cfg else None
        ),
        "stage_position_definition": (
            "generated_tokens_over_dense_trajectory_tokens_minus_one" if stage_cfg else None
        ),
        "operational_stage_counts": dict(sorted(stage_counts.items())),
        "trusted_stage_counts": dict(sorted(trusted_stage_counts.items())),
        "stage_probe_causal": (
            all(bool(row.get("stage_probe_causal", False)) for row in dense_rows)
            if stage_cfg
            else None
        ),
        "collection_config_fingerprint": config_fingerprint(
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
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    summary = validate_aligned_window_bank(cfg)
    write_json(cfg["paths"]["aligned_window_bank_validation"], summary)
    print(f"aligned-window-bank validation: {summary['status']}")
    for error in summary["errors"]:
        print(f"error: {error}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
