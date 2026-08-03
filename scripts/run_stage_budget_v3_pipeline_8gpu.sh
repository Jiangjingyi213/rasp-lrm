#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v3_smoke_8gpu.sh

smoke_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "08_budget_v3_actual_calibrated" / "00_smoke_actual_pruning" / "budget_v3_smoke_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${smoke_passed}" != "1" ]]; then
  echo "Budget v3 smoke did not reach 34% actual pruning; stopping before dev."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v3_dev_8gpu.sh

dev_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "08_budget_v3_actual_calibrated" / "01_dev_seed3" / "budget_v3_dev_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${dev_passed}" != "1" ]]; then
  echo "Budget v3 dev did not pass; stopping before seed42 confirmation."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v3_seed42_8gpu.sh
