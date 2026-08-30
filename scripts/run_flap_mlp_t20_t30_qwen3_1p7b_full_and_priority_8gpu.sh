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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
BASE_ROOT="${BASE_ROOT:-runs/12_additional_baselines/03_flap_mlp}"
LOG_ROOT="${LOG_ROOT:-logs/12_additional_baselines/03_flap_mlp}"
FLAP_RUN_FULL="${FLAP_RUN_FULL:-1}"
FLAP_RUN_PRIORITY="${FLAP_RUN_PRIORITY:-1}"
DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"

if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing FLAP calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi
export SOURCE_ROOT

mkdir -p "${BASE_ROOT}/00_preflight" "${LOG_ROOT}/00_preflight"

run_tier() {
  local tier="$1"
  local method
  local artifact
  local full_config
  local priority_config
  local full_root
  local priority_root
  local artifact_log

  case "${tier}" in
    t30)
      method="flap_mlp_t30_official"
      artifact="${BASE_ROOT}/00_preflight/flap_mlp_t30_artifact.json"
      full_config="configs/generated_additional_baselines/flap_mlp_t30_qwen3_1p7b_full.yaml"
      priority_config="configs/generated_additional_baselines/flap_mlp_t30_qwen3_1p7b_priority_suite.yaml"
      full_root="${BASE_ROOT}/01_t30_full_qwen3_1p7b"
      priority_root="${BASE_ROOT}/02_t30_priority_suite_qwen3_1p7b"
      ;;
    t20)
      method="flap_mlp_t20_official"
      artifact="${BASE_ROOT}/00_preflight/flap_mlp_t20_artifact.json"
      full_config="configs/generated_additional_baselines/flap_mlp_t20_qwen3_1p7b_full.yaml"
      priority_config="configs/generated_additional_baselines/flap_mlp_t20_qwen3_1p7b_priority_suite.yaml"
      full_root="${BASE_ROOT}/03_t20_full_qwen3_1p7b"
      priority_root="${BASE_ROOT}/04_t20_priority_suite_qwen3_1p7b"
      ;;
    *)
      echo "Unknown FLAP tier: ${tier}; expected t30 or t20." >&2
      exit 2
      ;;
  esac

  mkdir -p \
    "${LOG_ROOT}/00_preflight" \
    "${LOG_ROOT}/${tier}_full_qwen3_1p7b" \
    "${LOG_ROOT}/${tier}_priority_suite_qwen3_1p7b"

  artifact_log="${LOG_ROOT}/00_preflight/${tier}_prepare_artifact.log"
  if [[ "${FORCE_FLAP_PRECOMPUTE:-0}" == "1" || ! -f "${artifact}" ]]; then
    echo "==== START FLAP artifact ${tier}: ${artifact} ===="
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    "${PYTHON_BIN}" -m src.baselines.prepare_flap_mlp_artifact \
      --config "${full_config}" \
      --method "${method}" \
      --output "${artifact}" \
      > "${artifact_log}" 2>&1
    echo "DONE FLAP artifact ${tier}"
  else
    echo "SKIP FLAP artifact ${tier}; existing ${artifact}"
  fi

  if [[ "${FLAP_RUN_FULL}" == "1" ]]; then
    echo "==== START FLAP ${tier} full GSM8K + MATH500 ===="
    PYTHON="${PYTHON_BIN}" \
    CONFIG="${full_config}" \
    RUN_ROOT="${full_root}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    STAGE_SEED="${SEED}" \
    STAGE_FINAL_METHODS="${method}" \
    LOG_DIR="${LOG_ROOT}/${tier}_full_qwen3_1p7b" \
    RUN_LABEL="flap_mlp_${tier}_qwen3_1p7b_full" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    bash scripts/run_griffin_prompt_matched_full_gpu.sh
  fi

  if [[ "${FLAP_RUN_PRIORITY}" == "1" ]]; then
    echo "==== START FLAP ${tier} priority suite: ${DATASETS_OVERRIDE} ===="
    PYTHON="${PYTHON_BIN}" \
    CONFIG="${priority_config}" \
    RUN_ROOT="${priority_root}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    STAGE_SEED="${SEED}" \
    STAGE_FINAL_METHODS="${method}" \
    DATASETS_OVERRIDE="${DATASETS_OVERRIDE}" \
    LOG_DIR="${LOG_ROOT}/${tier}_priority_suite_qwen3_1p7b" \
    LOG_PREFIX="flap_mlp_${tier}_qwen3_1p7b_priority" \
    BASELINE_LABEL="FLAP-MLP ${tier} Qwen3-1.7B" \
    BASELINE_SCHEMA="flap_mlp_${tier}_qwen3_1p7b_priority_suite_aggregate_v1" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    bash scripts/run_griffin_prompt_priority_suite_gpu.sh
  fi

  echo "==== DONE FLAP ${tier} ===="
  echo "Full summary: ${full_root}/06_final/summary.json"
  echo "Priority summary: ${priority_root}/aggregate_summary.md"
}

read -r -a TIERS <<< "${FLAP_TIERS:-t30 t20}"
for tier in "${TIERS[@]}"; do
  run_tier "${tier}"
done

echo "==== ALL DONE FLAP-MLP tiers: ${TIERS[*]} ===="
