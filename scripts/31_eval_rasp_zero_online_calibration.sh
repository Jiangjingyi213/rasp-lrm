#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CONFIG_DIR="configs/generated_rasp_zero_online_calibration"

"${PYTHON}" scripts/31_prepare_rasp_zero_online_calibration_configs.py

while IFS= read -r config; do
  [[ -n "${config}" ]] || continue
  echo "START ${config}"
  "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
  echo "DONE ${config}"
done < <("${PYTHON}" - <<'PY'
import json
from pathlib import Path
manifest = json.load(open(Path("configs/generated_rasp_zero_online_calibration") / "manifest.json"))
for row in manifest:
    print(row["config"])
PY
)

"${PYTHON}" scripts/32_summarize_rasp_zero_online_calibration.py
