from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.utils.io import ensure_dir, read_json, read_jsonl, read_yaml, write_json, write_jsonl


RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def _boundary_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["id"]), int(row["boundary_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-dense-correct-per-source", type=int, default=100)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    for item in manifest:
        cfg = read_yaml(item["config"])
        data = cfg["data"]
        source = str(item["source"])
        if source == "gsm8k_train":
            valid = data.get("dataset") == "gsm8k" and data.get("split") == "train"
        elif source == "math_train":
            valid = (
                data.get("dataset") == "math500"
                and data.get("dataset_label") == "math_train"
                and data.get("name_or_path") == "rasbt/math_full_minus_math500"
                and data.get("split") == "train"
            )
        else:
            valid = False
        if not valid:
            raise ValueError(f"Action-risk source isolation check failed for {item['config']}")

    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(list)
    source_problems: dict[str, set[str]] = defaultdict(set)
    run_dirs = sorted(Path(item["run_dir"]) for item in manifest)
    if not run_dirs:
        raise ValueError(f"Manifest contains no bank shards: {args.manifest}")
    bank_root = Path(args.bank_root)
    if any(root.parent != bank_root for root in run_dirs):
        raise ValueError("Manifest run directories do not match --bank-root")
    for root in run_dirs:
        validation_path = root / "07_action_window_bank_validation.json"
        if not validation_path.exists():
            raise ValueError(f"Missing validation: {validation_path}")
        validation = read_json(validation_path)
        if (
            validation.get("status") != "ok"
            or validation.get("ratios") != RATIOS
            or validation.get("configured_max_boundaries_per_example") != 12
            or validation.get("boundary_sampling") != "uniform_full_trajectory"
        ):
            raise ValueError(f"{root}: expected validated action-risk pilot bank")
        rows = read_jsonl(root / "05_action_probe_dataset.jsonl")
        hidden = torch.load(root / "05_action_probe_hidden_states.pt", map_location="cpu")
        if len(rows) != len(hidden):
            raise ValueError(f"{root}: row/hidden mismatch")
        for index, row in enumerate(rows):
            grouped[_boundary_key(row)].append((row, hidden[index]))
            source_problems[str(row["dataset"])].add(str(row["id"]))

    insufficient = {
        source: len(problems)
        for source, problems in source_problems.items()
        if len(problems) < args.min_dense_correct_per_source
    }
    if insufficient:
        raise ValueError(
            "Action-risk pilot requires minimum dense-correct problems per source; "
            f"insufficient={insufficient}, required={args.min_dense_correct_per_source}"
        )
    unexpected_sources = set(source_problems) - {"gsm8k", "math_train"}
    if unexpected_sources or not {"gsm8k", "math_train"}.issubset(source_problems):
        raise ValueError(
            "Action-risk pilot must use only GSM8K train and isolated math_train sources; "
            f"observed={sorted(source_problems)}"
        )

    output_rows, output_hidden = [], []
    for key, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: float(item[0]["ratio"]))
        rows = [item[0] for item in candidates]
        ratios = [float(row["ratio"]) for row in rows]
        if ratios != RATIOS:
            raise ValueError(f"{key}: incomplete action grid {ratios}")
        if any(int(row["action_duration_tokens"]) < 16 for row in rows):
            continue
        reference = rows[0]
        output_rows.append(
            {
                "dataset": reference["dataset"],
                "id": reference["id"],
                "boundary_index": int(reference["boundary_index"]),
                "generated_tokens_at_boundary": int(reference["generated_tokens_at_boundary"]),
                "max_new_tokens": int(reference["max_new_tokens"]),
                "position": float(reference["position"]),
                "entropy": float(reference["entropy"]),
                "confidence": float(reference["confidence"]),
                "stage_probabilities": reference.get("stage_probabilities"),
                "candidate_ratios": RATIOS,
                "candidate_flipped": [bool(row["flipped"]) for row in rows],
                "candidate_token_divergence": [float(row["window_token_divergence"]) for row in rows],
            }
        )
        output_hidden.append(candidates[0][1])

    eligible_source_problems: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        eligible_source_problems[str(row["dataset"])].add(str(row["id"]))
    insufficient_eligible = {
        source: len(eligible_source_problems[source])
        for source in ("gsm8k", "math_train")
        if len(eligible_source_problems[source]) < args.min_dense_correct_per_source
    }
    if insufficient_eligible:
        raise ValueError(
            "Action-risk pilot requires full-window eligible dense-correct problems per source; "
            f"insufficient={insufficient_eligible}, required={args.min_dense_correct_per_source}"
        )

    output = ensure_dir(args.output_dir)
    write_jsonl(output / "01_action_risk_dataset.jsonl", output_rows)
    torch.save(torch.stack(output_hidden), output / "01_action_risk_hidden_states.pt")
    positives = Counter()
    totals = Counter()
    for row in output_rows:
        for ratio, flipped in zip(row["candidate_ratios"][1:], row["candidate_flipped"][1:]):
            positives[f"{ratio:.2f}"] += int(flipped)
            totals[f"{ratio:.2f}"] += 1
    write_json(
        output / "01_action_risk_data_summary.json",
        {
            "schema": "rasp_action_risk_pilot_v1",
            "source_dense_correct_problems": {
                source: len(problems) for source, problems in sorted(source_problems.items())
            },
            "source_full_window_eligible_dense_correct_problems": {
                source: len(problems)
                for source, problems in sorted(eligible_source_problems.items())
            },
            "minimum_dense_correct_per_source": args.min_dense_correct_per_source,
            "problem_count": len({(row["dataset"], row["id"]) for row in output_rows}),
            "boundary_count": len(output_rows),
            "ratios": RATIOS,
            "positive_flips_by_ratio": dict(positives),
            "action_rows_by_ratio": dict(totals),
            "math_training_source_isolated_from_math500": "math_train" in source_problems,
            "source_manifest": args.manifest,
            "causal_model_position_definition": "generated_tokens_at_boundary/max_new_tokens",
            "stage_probability_status": (
                "not_used_by_pilot_models: current S1 stage probability depends on "
                "full-trajectory-relative position and is not a causal online feature"
            ),
        },
    )


if __name__ == "__main__":
    main()
