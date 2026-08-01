#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_output_aware_phase_a_8gpu.sh

phase_a_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "01_budget_only_dev" / "phase_a_selection.json"
print("1" if json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_output_aware_phase_b_8gpu.sh

phase_b_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "02_output_aware_only_dev" / "phase_b_selection.json"
print("1" if json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${phase_a_passed}" != "1" || "${phase_b_passed}" != "1" ]]; then
  echo "Phase A or Phase B did not pass; stopping before combined run."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_output_aware_phase_c_8gpu.sh
