#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"

smoke_selection="${RUN_ROOT}/10_budget_v41_stage_lift_32p/00_smoke_32p/budget_v41_smoke_selection.json"
if [[ -f "${smoke_selection}" ]]; then
  existing_smoke_passed="$(SELECTION="${smoke_selection}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["SELECTION"])
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
else
  existing_smoke_passed="0"
fi

if [[ "${existing_smoke_passed}" == "1" ]]; then
  echo "Reusing passed Budget v4.1 smoke selection: ${smoke_selection}"
else
  RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_smoke_8gpu.sh
fi

smoke_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "00_smoke_32p" / "budget_v41_smoke_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${smoke_passed}" != "1" ]]; then
  echo "Budget v4.1 smoke did not reach 32.0%-34.0% actual pruning; stopping before dev."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_dev_8gpu.sh

dev_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "01_dev_seed3" / "budget_v41_dev_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${dev_passed}" != "1" ]]; then
  echo "Budget v4.1 dev did not pass; stopping before seed42 confirmation."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_seed42_8gpu.sh

seed42_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "02_seed42_confirmation" / "budget_v41_seed42_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${seed42_passed}" != "1" ]]; then
  echo "Budget v4.1 seed42 confirmation did not pass; stopping before full."
  exit 0
fi

compare_output="${RUN_ROOT}/10_budget_v41_stage_lift_32p/02_seed42_confirmation/budget_v41_seed3_vs_seed42_compare.json"
RUN_ROOT="${RUN_ROOT}" COMPARE_OUTPUT="${compare_output}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
out_path = Path(os.environ["COMPARE_OUTPUT"])
dev_selection_path = run_root / "10_budget_v41_stage_lift_32p" / "01_dev_seed3" / "budget_v41_dev_selection.json"
seed42_selection_path = run_root / "10_budget_v41_stage_lift_32p" / "02_seed42_confirmation" / "budget_v41_seed42_selection.json"

def load_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))

dev_selection = load_json(dev_selection_path)
seed42_selection = load_json(seed42_selection_path)
method = dev_selection.get("selected_candidate_method")

def score(selection: dict) -> dict:
    rows = []
    for candidate in selection.get("candidates", []):
        if candidate.get("method") != method:
            continue
        rows = candidate.get("datasets", [])
        break
    avg_accuracy = sum(row.get("accuracy", 0.0) for row in rows) / len(rows) if rows else None
    avg_protocol = sum(row.get("protocol_valid_accuracy", row.get("accuracy", 0.0)) for row in rows) / len(rows) if rows else None
    avg_pruning = sum(row.get("actual_pruning", 0.0) for row in rows) / len(rows) if rows else None
    return {
        "phase": selection.get("phase"),
        "phase_passed": selection.get("phase_passed"),
        "seed": selection.get("seed"),
        "avg_accuracy": avg_accuracy,
        "avg_protocol_valid_accuracy": avg_protocol,
        "avg_actual_pruning": avg_pruning,
        "datasets": rows,
    }

seed3 = score(dev_selection)
seed42 = score(seed42_selection)
eligible = [row for row in [seed3, seed42] if row.get("phase_passed")]
selected_seed = None
if eligible:
    selected_seed = max(
        eligible,
        key=lambda row: (
            row.get("avg_protocol_valid_accuracy") if row.get("avg_protocol_valid_accuracy") is not None else -1.0,
            row.get("avg_accuracy") if row.get("avg_accuracy") is not None else -1.0,
            row.get("avg_actual_pruning") if row.get("avg_actual_pruning") is not None else -1.0,
        ),
    ).get("seed")

out = {
    "schema": "stage_budget_v41_seed_compare",
    "selected_method": method,
    "selected_full_seed": selected_seed,
    "selection_rule": "Choose the passed seed with highest average protocol-valid accuracy, then accuracy, then actual pruning.",
    "seed3": seed3,
    "seed42": seed42,
    "full_not_run_by_pipeline": True,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Budget v4.1 seed comparison written to {out_path}; full is intentionally not started.")
if selected_seed is not None:
    print(f"Recommended full seed: {selected_seed}")
PY

echo "Budget v4.1 pipeline stopped after seed42 confirmation by design. Inspect ${compare_output} before launching full."
