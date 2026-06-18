#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG="${CONFIG:-configs/stage_calibrated_pruning/main.yaml}"
PROFILE="${PROFILE:-smoke}"
run_stage() {
  local stage="$1"
  echo "START stage=${stage} profile=${PROFILE}"
  "${PYTHON}" -m src.main_stage_calibrated_pruning \
      --config "${CONFIG}" \
      --profile "${PROFILE}" \
      --stage "${stage}" || {
    local status=$?
    echo "FAILED stage=${stage}" >&2
    return "${status}"
  }
  echo "DONE stage=${stage}"
}

finalize_partial() {
  "${PYTHON}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG}" \
    --profile "${PROFILE}" \
    --stage summarize \
    --force >/dev/null 2>&1 || true
}

trap finalize_partial ERR

run_stage preflight

while true; do
  run_stage build_pool
  run_stage generate_trajectories
  if run_stage select_trajectories; then
    break
  else
    status=$?
  fi
  if [[ "${PROFILE}" != "formal" || "${status}" -ne 42 ]]; then
    exit "${status}"
  fi
  echo "Formal selection requested another 400 candidate problems; expanding."
done

for stage in calibrate_masks validate_masks evaluate_dev evaluate_final summarize; do
  run_stage "${stage}"
done
