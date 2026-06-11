#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
PATTERN="${1:-runs/eval_*_qwen3_*_budget}"

"$PYTHON" - "$PATTERN" <<'PY'
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

pattern = sys.argv[1]

def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def parse_run_name(name: str) -> tuple[str, str]:
    # eval_dense_qwen3_gsm8k_budget
    # eval_griffin_p20_qwen3_math500_budget
    parts = name.split("_qwen3_")
    method = parts[0].removeprefix("eval_")
    dataset = parts[1].removesuffix("_budget") if len(parts) > 1 else "unknown"
    return method, dataset

runs = sorted(Path(p) for p in glob.glob(pattern))
print("run,method,dataset,total,correct,accuracy")
for run_dir in runs:
    rows = load_rows(run_dir / "01_trajectories.jsonl")
    total = len(rows)
    correct = sum(bool(r.get("correct")) for r in rows)
    acc = correct / total if total else 0.0
    method, dataset = parse_run_name(run_dir.name)
    print(f"{run_dir},{method},{dataset},{total},{correct},{acc:.6f}")

print()
for dataset in sorted({parse_run_name(p.name)[1] for p in runs}):
    dense_dir = Path(f"runs/02_baselines/eval_dense_qwen3_{dataset}_budget")
    dense_rows = {r["id"]: bool(r.get("correct")) for r in load_rows(dense_dir / "01_trajectories.jsonl")}
    if not dense_rows:
        continue
    print(f"paired_dataset={dataset}")
    print("method,shared,both_correct,dense_correct_pruned_wrong,dense_wrong_pruned_correct,both_wrong")
    for run_dir in runs:
        method, run_dataset = parse_run_name(run_dir.name)
        if run_dataset != dataset or method == "dense":
            continue
        pruned_rows = {r["id"]: bool(r.get("correct")) for r in load_rows(run_dir / "01_trajectories.jsonl")}
        ids = sorted(set(dense_rows) & set(pruned_rows))
        both_correct = sum(dense_rows[i] and pruned_rows[i] for i in ids)
        flip_wrong = sum(dense_rows[i] and not pruned_rows[i] for i in ids)
        rescue = sum((not dense_rows[i]) and pruned_rows[i] for i in ids)
        both_wrong = sum((not dense_rows[i]) and (not pruned_rows[i]) for i in ids)
        print(f"{method},{len(ids)},{both_correct},{flip_wrong},{rescue},{both_wrong}")
    print()
PY
