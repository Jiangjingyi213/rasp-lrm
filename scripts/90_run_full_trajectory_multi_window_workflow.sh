#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FULL_TRAJECTORY_ROOT="${FULL_TRAJECTORY_ROOT:-runs/07_stage_aware/10_full_trajectory_multi_window}"
LOG_ROOT="${FULL_TRAJECTORY_LOG_ROOT:-logs/07_stage_aware/10_full_trajectory_multi_window}"
export PYTHON GPU_IDS FULL_TRAJECTORY_ROOT
mkdir -p "${FULL_TRAJECTORY_ROOT}" "${LOG_ROOT}"

finalize() {
  local failed_stage="${1:-}"
  local active_stage="${2:-}"
  args=(--root "${FULL_TRAJECTORY_ROOT}")
  [[ -z "${failed_stage}" ]] || args+=(--failed-stage "${failed_stage}")
  [[ -z "${active_stage}" ]] || args+=(--active-stage "${active_stage}")
  "${PYTHON}" -m src.main_summarize_full_trajectory_workflow "${args[@]}"
}

run_stage() {
  local stage="$1"
  shift
  finalize "" "${stage}"
  echo "START stage=${stage}"
  if "$@" > "${LOG_ROOT}/${stage}.log" 2>&1; then
    echo "DONE stage=${stage}"
    finalize "" "${stage}"
  else
    status=$?
    echo "FAILED stage=${stage}; inspect ${LOG_ROOT}/${stage}.log" >&2
    finalize "${stage}" "${stage}"
    exit "${status}"
  fi
}

run_stage code_preflight_tests \
  "${PYTHON}" -m unittest discover -s tests

run_stage existing_bank_preflight \
  "${PYTHON}" -m src.main_preflight_full_trajectory \
  --output-dir "${FULL_TRAJECTORY_ROOT}/00_preflight"

run_stage dense_bank_smoke_collect \
  env PROFILE=dense_smoke LOG_DIR="${LOG_ROOT}/dense_smoke" \
  bash scripts/84_collect_full_trajectory_bank.sh

run_stage dense_bank_smoke_gate \
  env PROFILE=dense_smoke bash scripts/85_prepare_analyze_full_trajectory_bank.sh

run_stage dense_bank_pilot_collect \
  env PROFILE=dense_pilot LOG_DIR="${LOG_ROOT}/dense_pilot" \
  bash scripts/84_collect_full_trajectory_bank.sh

run_stage dense_bank_pilot_gate \
  env PROFILE=dense_pilot bash scripts/85_prepare_analyze_full_trajectory_bank.sh

run_stage fixed_multi_window_dev \
  env LOG_DIR="${LOG_ROOT}/fixed_multi_window_dev" \
  bash scripts/87_eval_fixed_multi_window_dev.sh

run_stage on_policy_smoke \
  env LOG_DIR="${LOG_ROOT}/on_policy_smoke" \
  bash scripts/89_collect_on_policy_smoke.sh

finalize
echo "Workflow complete. See ${FULL_TRAJECTORY_ROOT}/final_workflow_report_zh.md"
