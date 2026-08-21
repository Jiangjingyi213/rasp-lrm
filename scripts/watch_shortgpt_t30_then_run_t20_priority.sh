#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

T30_ROOT="${T30_ROOT:-runs/12_additional_baselines/01_shortgpt_t30/02_priority_suite_qwen3_1p7b}"
T20_LOG_ROOT="${T20_LOG_ROOT:-logs/12_additional_baselines/02_shortgpt_t20}"
WATCH_LOG="${WATCH_LOG:-${T20_LOG_ROOT}/watch_t30_then_t20.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DATASETS="${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge}"

mkdir -p "${T20_LOG_ROOT}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${WATCH_LOG}"
}

priority_complete() {
  [[ -f "${T30_ROOT}/aggregate_summary.json" ]] || return 1
  local dataset
  for dataset in ${DATASETS}; do
    [[ -f "${T30_ROOT}/${dataset}/06_final/summary.json" ]] || return 1
  done
  return 0
}

log "Watching ShortGPT T30 priority completion: ${T30_ROOT}"
log "Required datasets: ${DATASETS}"
while ! priority_complete; do
  existing="$(find "${T30_ROOT}" -path '*/06_final/summary.json' -print 2>/dev/null | sort | tr '\n' ' ')"
  log "T30 not complete yet; existing summaries: ${existing:-none}; sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "T30 priority is complete. Launching ShortGPT T20 full + priority."
exec env DATASETS_OVERRIDE="${DATASETS}" bash scripts/run_shortgpt_t20_qwen3_1p7b_full_and_priority.sh
