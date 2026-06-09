from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from src.probes.rasp_train_dataset import DEFAULT_RATIOS
from src.rasp.phase_b2 import boundary_key, stratified_problem_split
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    grouped = defaultdict(list)
    hidden_by_boundary = {}
    filtered_incomplete = set()
    for run_dir in args.run_dirs:
        root = Path(run_dir)
        validation = read_json(root / "07_aligned_window_bank_validation.json")
        if (
            validation.get("status") != "ok"
            or validation.get("action_window_alignment") != "affected_next_token_decisions_v2"
            or validation.get("configured_window_tokens") != 16
            or validation.get("configured_max_boundaries_per_example") != 12
        ):
            raise ValueError(f"{root}: expected a validated 16-token/12-window aligned bank shard")
        rows = read_jsonl(root / "05_probe_dataset.jsonl")
        hidden = torch.load(root / "05_probe_hidden_states.pt", map_location="cpu")
        if len(rows) != len(hidden):
            raise ValueError(f"{root}: row/hidden count mismatch")
        for index, row in enumerate(rows):
            key = boundary_key(row)
            if int(row["action_duration_tokens"]) < int(row["window_tokens"]):
                filtered_incomplete.add(key)
            grouped[key].append(row)
            current_hidden = hidden[index]
            if key in hidden_by_boundary and not torch.equal(hidden_by_boundary[key], current_hidden):
                raise ValueError(f"{key}: candidate actions do not share the same pre-action boundary hidden state")
            hidden_by_boundary[key] = current_hidden
    output_rows, output_hidden = [], []
    for key, values in sorted(grouped.items()):
        if key in filtered_incomplete:
            continue
        ordered = sorted(values, key=lambda row: float(row["ratio"]))
        ratios = [float(row["ratio"]) for row in ordered]
        if ratios != DEFAULT_RATIOS:
            raise ValueError(f"{key}: incomplete or unexpected ratio grid {ratios}")
        reference = ordered[0]
        for row in ordered[1:]:
            for field in ("entropy", "confidence", "position", "generated_tokens_at_boundary"):
                if abs(float(row[field]) - float(reference[field])) > 1e-9:
                    raise ValueError(f"{key}: candidate actions disagree on pre-action field {field}")
        output_rows.append(
            {
                "dataset": reference.get("dataset"),
                "id": reference["id"],
                "boundary_index": int(reference["boundary_index"]),
                "segment_id": int(reference["boundary_index"]),
                "segment_index": int(reference["boundary_index"]),
                "generated_tokens_at_boundary": int(reference["generated_tokens_at_boundary"]),
                "position": float(reference["position"]),
                "entropy": float(reference["entropy"]),
                "confidence": float(reference["confidence"]),
                "candidate_ratios": ratios,
                "candidate_flipped": [bool(row["flipped"]) for row in ordered],
                "candidate_unsafe": [bool(row["flipped"]) for row in ordered],
                "candidate_token_divergence": [float(row["window_token_divergence"]) for row in ordered],
                "candidate_hidden_cosine_distance": [
                    max(0.0, min(1.0, float(row["window_end_hidden_cosine_distance"])))
                    for row in ordered
                ],
            }
        )
        output_hidden.append(hidden_by_boundary[key])
    output = ensure_dir(args.output_dir)
    write_jsonl(output / "01_phase_b2_dataset.jsonl", output_rows)
    torch.save(torch.stack(output_hidden), output / "01_phase_b2_hidden_states.pt")
    for seed in args.seeds:
        write_json(output / f"split_seed_{seed}.json", stratified_problem_split(output_rows, seed))
    write_json(
        output / "01_phase_b2_data_summary.json",
        {
            "source_run_dirs": args.run_dirs,
            "problem_count": len({(row["dataset"], row["id"]) for row in output_rows}),
            "boundary_count": len(output_rows),
            "filtered_incomplete_boundaries": len(filtered_incomplete),
            "nonzero_action_rows": len(output_rows) * (len(DEFAULT_RATIOS) - 1),
            "nonzero_flip_positives": sum(
                int(value) for row in output_rows for value in row["candidate_flipped"][1:]
            ),
        },
    )


if __name__ == "__main__":
    main()
