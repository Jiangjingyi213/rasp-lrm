from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.utils.io import ensure_dir, read_json, read_jsonl, read_yaml, write_json, write_jsonl


RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BOUNDARIES = [32, 96, 160]


def problem_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def boundary_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (*problem_key(row), int(row["generated_tokens_at_boundary"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-complete-problems-per-source", type=int, default=100)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    bank_root = Path(args.bank_root)
    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(list)
    observed_problems: dict[str, set[str]] = defaultdict(set)
    for item in manifest:
        cfg = read_yaml(item["config"])
        bank_cfg = cfg["aligned_window_bank"]
        if [float(value) for value in bank_cfg["ratios"]] != RATIOS:
            raise ValueError(f"{item['config']}: unexpected ratio grid")
        if [int(value) for value in bank_cfg.get("boundary_positions", [])] != BOUNDARIES:
            raise ValueError(f"{item['config']}: expected exact boundary positions {BOUNDARIES}")
        if not cfg.get("stage_sensitivity"):
            raise ValueError(f"{item['config']}: stage sensitivity annotation is required")
        root = Path(item["run_dir"])
        if root.parent != bank_root:
            raise ValueError("Manifest run directories do not match --bank-root")
        validation = read_json(root / "07_stage_action_bank_validation.json")
        if (
            validation.get("status") != "ok"
            or validation.get("configured_boundary_positions") != BOUNDARIES
            or not validation.get("stage_sensitivity_enabled")
        ):
            raise ValueError(f"{root}: expected validated exact-boundary stage-action bank")
        rows = read_jsonl(root / "05_stage_action_probe_dataset.jsonl")
        hidden = torch.load(root / "05_stage_action_probe_hidden_states.pt", map_location="cpu")
        if len(rows) != len(hidden):
            raise ValueError(f"{root}: row/hidden mismatch")
        for index, row in enumerate(rows):
            grouped[boundary_key(row)].append((row, hidden[index]))
            observed_problems[str(row["dataset"])].add(str(row["id"]))

    by_problem: dict[tuple[str, str], set[int]] = defaultdict(set)
    valid_boundaries: dict[tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for key, candidates in grouped.items():
        candidates.sort(key=lambda item: float(item[0]["ratio"]))
        ratios = [float(item[0]["ratio"]) for item in candidates]
        if ratios != RATIOS:
            raise ValueError(f"{key}: incomplete action grid {ratios}")
        if any(int(item[0]["action_duration_tokens"]) < 16 for item in candidates):
            continue
        valid_boundaries[key] = candidates
        by_problem[key[:2]].add(key[2])
    complete_problems = {
        key for key, positions in by_problem.items() if positions == set(BOUNDARIES)
    }
    complete_by_source = Counter(key[0] for key in complete_problems)
    insufficient = {
        source: complete_by_source[source]
        for source in ("gsm8k", "math_train")
        if complete_by_source[source] < args.min_complete_problems_per_source
    }
    if insufficient:
        raise ValueError(
            "Stage-Action-Risk v2 requires complete exact-boundary problems; "
            f"insufficient={insufficient}, required={args.min_complete_problems_per_source}"
        )

    output_rows, output_hidden = [], []
    for key in sorted(valid_boundaries):
        if key[:2] not in complete_problems:
            continue
        candidates = valid_boundaries[key]
        rows = [item[0] for item in candidates]
        dense = rows[0]
        probabilities = dense.get("stage_probabilities")
        if dense["operational_stage"] == "verification":
            probabilities = {"setup": 0.0, "reasoning": 0.0, "final": 0.0}
        elif not isinstance(probabilities, dict) or set(probabilities) != {"setup", "reasoning", "final"}:
            raise ValueError(f"{key}: missing three-class stage probabilities")
        output_rows.append(
            {
                "dataset": dense["dataset"],
                "id": dense["id"],
                "boundary_index": BOUNDARIES.index(key[2]),
                "generated_tokens_at_boundary": key[2],
                "max_new_tokens": int(dense["max_new_tokens"]),
                "entropy": float(dense["entropy"]),
                "confidence": float(dense["confidence"]),
                "operational_stage": dense["operational_stage"],
                "stage_source": dense["stage_source"],
                "reasoning_accepted": bool(dense["reasoning_accepted"]),
                "stage_confidence": float(dense["stage_confidence"]),
                "stage_probabilities": {
                    stage: float(probabilities[stage]) for stage in ("setup", "reasoning", "final")
                },
                "candidate_ratios": RATIOS,
                "candidate_flipped": [bool(row["flipped"]) for row in rows],
                "candidate_token_divergence": [
                    float(row["window_token_divergence"]) for row in rows
                ],
            }
        )
        output_hidden.append(candidates[0][1])

    output = ensure_dir(args.output_dir)
    write_jsonl(output / "01_stage_action_risk_dataset.jsonl", output_rows)
    torch.save(torch.stack(output_hidden), output / "01_stage_action_risk_hidden_states.pt")
    stage_counts = Counter(str(row["operational_stage"]) for row in output_rows)
    boundary_stage_counts = Counter(
        f"{int(row['generated_tokens_at_boundary'])}|{row['operational_stage']}"
        for row in output_rows
    )
    stage_ratio_flips = Counter()
    stage_ratio_rows = Counter()
    for row in output_rows:
        for ratio, flipped in zip(row["candidate_ratios"][1:], row["candidate_flipped"][1:]):
            label = f"{row['operational_stage']}|{float(ratio):.2f}"
            stage_ratio_rows[label] += 1
            stage_ratio_flips[label] += int(flipped)
    write_json(
        output / "01_stage_action_risk_data_summary.json",
        {
            "schema": "rasp_stage_action_risk_exact_boundaries_v2",
            "required_boundaries": BOUNDARIES,
            "ratios": RATIOS,
            "observed_dense_correct_problems": {
                source: len(values) for source, values in sorted(observed_problems.items())
            },
            "complete_exact_boundary_problems": dict(sorted(complete_by_source.items())),
            "minimum_complete_problems_per_source": args.min_complete_problems_per_source,
            "problem_count": len(complete_problems),
            "boundary_count": len(output_rows),
            "stage_counts": dict(sorted(stage_counts.items())),
            "boundary_stage_counts": [
                {
                    "boundary": int(label.split("|")[0]),
                    "stage": label.split("|")[1],
                    "rows": count,
                }
                for label, count in sorted(boundary_stage_counts.items())
            ],
            "stage_ratio_summary": [
                {
                    "stage": label.split("|")[0],
                    "ratio": float(label.split("|")[1]),
                    "rows": stage_ratio_rows[label],
                    "flips": stage_ratio_flips[label],
                    "flip_rate": stage_ratio_flips[label] / stage_ratio_rows[label],
                }
                for label in sorted(stage_ratio_rows)
            ],
            "note": (
                "Every retained problem contains all exact 32/96/160 boundaries. "
                "Stage probabilities are action-preceding annotations; no controller is trained yet."
            ),
        },
    )


if __name__ == "__main__":
    main()
