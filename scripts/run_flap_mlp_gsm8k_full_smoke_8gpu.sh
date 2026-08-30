#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
BASE_ROOT="runs/12_additional_baselines/03_flap_mlp"
LOG_ROOT="logs/12_additional_baselines/03_flap_mlp"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"

if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing FLAP calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi
export SOURCE_ROOT

mkdir -p "${BASE_ROOT}/00_preflight" "${LOG_ROOT}/00_preflight" "${LOG_ROOT}/05_gsm8k_full_smoke"

prepare_artifact() {
  local tier="$1"
  local method="$2"
  local config="$3"
  local artifact="${BASE_ROOT}/00_preflight/flap_mlp_${tier}_artifact.json"
  local log_path="${LOG_ROOT}/00_preflight/${tier}_prepare_artifact.log"
  if [[ "${FORCE_FLAP_PRECOMPUTE:-0}" == "1" || ! -f "${artifact}" ]]; then
    echo "START FLAP ${tier} artifact: ${artifact}"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    "${PYTHON_BIN}" -m src.baselines.prepare_flap_mlp_artifact \
      --config "${config}" \
      --method "${method}" \
      --output "${artifact}" \
      > "${log_path}" 2>&1
    echo "DONE FLAP ${tier} artifact"
  else
    echo "SKIP FLAP ${tier} artifact; existing ${artifact}"
  fi
}

run_smoke() {
  local tier="$1"
  local method="$2"
  local config="$3"
  local run_root="$4"
  prepare_artifact "${tier}" "${method}" "${config}"
  echo "START FLAP ${tier} GSM8K full smoke; run_root=${run_root}"
  PYTHON="${PYTHON_BIN}" \
  CONFIG="${config}" \
  RUN_ROOT="${run_root}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  STAGE_SEED="${SEED}" \
  STAGE_FINAL_METHODS="${method}" \
  LOG_DIR="${LOG_ROOT}/05_gsm8k_full_smoke" \
  RUN_LABEL="flap_mlp_${tier}_qwen3_1p7b_gsm8k_full_smoke" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  bash scripts/run_griffin_prompt_matched_full_gpu.sh
}

read -r -a TIERS <<< "${FLAP_SMOKE_TIERS:-t30}"
for tier in "${TIERS[@]}"; do
  case "${tier}" in
    t30)
      run_smoke \
        "t30" \
        "flap_mlp_t30_official" \
        "configs/generated_additional_baselines/flap_mlp_t30_qwen3_1p7b_gsm8k_smoke.yaml" \
        "${BASE_ROOT}/05_t30_gsm8k_full_smoke"
      ;;
    t20)
      run_smoke \
        "t20" \
        "flap_mlp_t20_official" \
        "configs/generated_additional_baselines/flap_mlp_t20_qwen3_1p7b_gsm8k_smoke.yaml" \
        "${BASE_ROOT}/06_t20_gsm8k_full_smoke"
      ;;
    *)
      echo "Unknown FLAP_SMOKE_TIERS value: ${tier}; expected t30 or t20." >&2
      exit 2
      ;;
  esac
done

echo "ALL DONE FLAP GSM8K smoke tiers: ${TIERS[*]}"
