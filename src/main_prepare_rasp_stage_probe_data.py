from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.rasp.stage_probe import (
    STAGES,
    STAGE_PROBE_SCHEMA,
    classify_operational_stage,
    problem_stage_split,
)
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--exclude-audit-labels")
    parser.add_argument("--min-stage-rows", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    deduplicated: dict[tuple[str, str, int], tuple[dict[str, Any], torch.Tensor]] = {}
    verification_overrides: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_counts: Counter[tuple[str, str, int]] = Counter()
    source_stage_mismatches = 0
    for run_dir in args.run_dirs:
        root = Path(run_dir)
        rows = read_jsonl(root / "05_probe_dataset.jsonl")
        hidden = torch.load(root / "05_probe_hidden_states.pt", map_location="cpu")
        if len(rows) != len(hidden):
            raise ValueError(f"Rows and hidden states differ under {root}")
        for index, row in enumerate(rows):
            hidden_index = int(row.get("hidden_index", index))
            if hidden_index < 0 or hidden_index >= len(hidden):
                raise ValueError(f"Invalid hidden_index={hidden_index} under {root}")
            num_segments = max(1, int(row.get("num_segments", 1)))
            segment_index = int(row.get("segment_index", row["segment_id"]))
            segment_text = str(row.get("segment_text", ""))
            source_stage = str(row.get("segment_type", "unknown"))
            stage = classify_operational_stage(segment_text, segment_index, num_segments)
            key = (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))
            if stage == "verification":
                verification_overrides.setdefault(
                    key,
                    {
                        "dataset": key[0],
                        "id": key[1],
                        "segment_id": key[2],
                        "segment_index": segment_index,
                        "num_segments": num_segments,
                        "segment_text": segment_text,
                        "runtime_action": "dense_override",
                    },
                )
                continue
            if stage not in STAGES:
                continue
            if stage != source_stage:
                source_stage_mismatches += 1
            duplicate_counts[key] += 1
            if key in deduplicated:
                previous_row, previous_hidden = deduplicated[key]
                for field in ("segment_text", "entropy", "confidence"):
                    if previous_row.get(field) != row.get(field):
                        raise ValueError(f"Inconsistent duplicate stage field {field} for {key}")
                if not torch.equal(previous_hidden, hidden[hidden_index]):
                    raise ValueError(f"Inconsistent duplicate hidden state for {key}")
                continue
            output_row = {
                "dataset": key[0],
                "id": key[1],
                "segment_id": key[2],
                "segment_index": segment_index,
                "num_segments": num_segments,
                "position": segment_index / max(1, num_segments - 1),
                "stage": stage,
                "source_stage": source_stage,
                "segment_text": segment_text,
                "entropy": float(row.get("entropy", 0.0)),
                "confidence": float(row.get("confidence", 0.0)),
                "source_run_dir": str(root),
                "source_hidden_index": hidden_index,
            }
            # Clone each retained segment so the deduplicated map does not keep
            # the full action-repeated source tensor storage alive.
            deduplicated[key] = (
                output_row,
                hidden[hidden_index].detach().float().cpu().clone(),
            )

    ordered = [deduplicated[key] for key in sorted(deduplicated)]
    rows = [row for row, _hidden in ordered]
    hidden = torch.stack([value for _row, value in ordered])
    stage_counts = Counter(row["stage"] for row in rows)
    insufficient = {
        stage: stage_counts[stage]
        for stage in STAGES
        if stage_counts[stage] < args.min_stage_rows
    }
    if insufficient:
        raise ValueError(
            f"Insufficient rows for reliable stage probe: {insufficient}; "
            f"required at least {args.min_stage_rows} per stage"
        )
    output = ensure_dir(args.output_dir)
    write_jsonl(output / "01_stage_dataset.jsonl", rows)
    write_jsonl(
        output / "01_verification_dense_overrides.jsonl",
        [verification_overrides[key] for key in sorted(verification_overrides)],
    )
    torch.save(hidden, output / "01_stage_hidden_states.pt")
    for seed in args.seeds:
        write_json(output / f"split_seed_{seed}.json", problem_stage_split(rows, seed))

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded_audit_keys: set[tuple[str, str, int]] = set()
    if args.exclude_audit_labels:
        with Path(args.exclude_audit_labels).open(newline="", encoding="utf-8") as handle:
            excluded_audit_keys = {
                (str(row["dataset"]), str(row["id"]), int(row["segment_id"]))
                for row in csv.DictReader(handle)
            }
    for row in rows:
        key = (str(row["dataset"]), str(row["id"]), int(row["segment_id"]))
        if key not in excluded_audit_keys:
            by_stratum[(str(row["dataset"]), str(row["stage"]))].append(row)
    audit = []
    rng = random.Random(1)
    strata = sorted(by_stratum)
    per_stratum = max(1, args.audit_size // max(1, len(strata)))
    for stratum in strata:
        values = sorted(by_stratum[stratum], key=lambda row: (row["id"], row["segment_id"]))
        rng.shuffle(values)
        for row in values[:per_stratum]:
            audit.append(
                {
                    **row,
                    "audited_stage": "",
                    "audit_status": "",
                    "audit_note": "",
                }
            )
    remaining = args.audit_size - len(audit)
    if remaining > 0:
        used = {(row["dataset"], row["id"], row["segment_id"]) for row in audit}
        candidates = [
            row
            for row in rows
            if (row["dataset"], row["id"], row["segment_id"]) not in used
            and (row["dataset"], row["id"], row["segment_id"]) not in excluded_audit_keys
        ]
        rng.shuffle(candidates)
        audit.extend(
            {**row, "audited_stage": "", "audit_status": "", "audit_note": ""}
            for row in candidates[:remaining]
        )
    write_jsonl(output / "02_stage_manual_audit.jsonl", audit[: args.audit_size])
    audit_rows = audit[: args.audit_size]
    if not audit_rows:
        raise ValueError("No valid stage rows available for manual audit")
    with (output / "02_stage_manual_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    write_json(
        output / "00_stage_data_summary.json",
        {
            "schema": STAGE_PROBE_SCHEMA,
            "rows": len(rows),
            "problems": len({(row["dataset"], row["id"]) for row in rows}),
            "stage_counts": dict(stage_counts),
            "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
            "duplicate_action_rows_per_segment": {
                "min": min(duplicate_counts.values()),
                "max": max(duplicate_counts.values()),
            },
            "source_stage_mismatch_action_rows": source_stage_mismatches,
            "verification_dense_override_rows": len(verification_overrides),
            "excluded_prior_audit_rows": len(excluded_audit_keys),
            "audit_rows": len(audit_rows),
        },
    )


if __name__ == "__main__":
    main()
