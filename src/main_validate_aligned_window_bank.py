from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
    if len(rows) != len(probe_rows):
        errors.append("Counterfactual and probe row counts differ")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
        if row.get("action_scope") != "single_fixed_window_then_dense":
            errors.append("Aligned bank contains a non-window action scope")
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
    max_replay_flip_rate = float(cfg.get("max_dense_replay_flip_rate", 0.05))
    if replay_flip_rate > max_replay_flip_rate:
        errors.append(
            f"Dense replay flip rate is high: {replay_flip_rate:.4f} > {max_replay_flip_rate:.4f}"
        )
    ratio_counts = Counter(f"{float(row['ratio']):.2f}" for row in rows)
    token_sources = sorted({str(row.get("boundary_token_source")) for row in rows})
    if token_sources != ["trajectory_generated_token_ids"]:
        errors.append(f"Formal aligned bank must use original generated token IDs, got {token_sources}")
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "boundaries": len(grouped),
        "counterfactual_rows": len(rows),
        "ratios": ratios,
        "ratio_counts": dict(sorted(ratio_counts.items())),
        "dense_control_paired_flip_rate": dense_flip_rate,
        "dense_replay_flip_rate_from_baseline": replay_flip_rate,
        "configured_window_tokens": int(cfg.get("window_tokens", 16)),
        "configured_max_boundaries_per_example": cfg.get("max_boundaries_per_example"),
        "action_scope": "single_fixed_window_then_dense",
        "ranking_scope": "initial_prompt_prefill_fixed",
        "boundary_token_sources": token_sources,
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
