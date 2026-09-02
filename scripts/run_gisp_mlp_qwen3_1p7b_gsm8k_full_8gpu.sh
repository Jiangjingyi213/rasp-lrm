#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/generated_additional_baselines/gisp_mlp_t20p22_mixed_qwen3_1p7b_gsm8k_full.yaml}"
PROFILE="${PROFILE:-pilot}"
RUN_ROOT="${RUN_ROOT:-runs/12_additional_baselines/04_gisp_mlp/04_t20p22_mixed_gsm8k_full}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-gisp_mlp_t20p22_mixed}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/04_gisp_mlp}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
RUN_LABEL="${RUN_LABEL:-gisp_mlp_t20p22_mixed_qwen3_1p7b_gsm8k_full}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
C4_CALIBRATION_PATH="${C4_CALIBRATION_PATH:-runs/12_additional_baselines/04_gisp_mlp/02_t35_c4_gsm8k_full/00_c4_calibration/c4_2000_seed3.jsonl}"
C4_CALIBRATION_SAMPLES="${C4_CALIBRATION_SAMPLES:-2000}"
C4_CALIBRATION_MIN_CHARS="${C4_CALIBRATION_MIN_CHARS:-64}"
C4_CALIBRATION_BUFFER_SIZE="${C4_CALIBRATION_BUFFER_SIZE:-10000}"
GISP_BUILD_C4="${GISP_BUILD_C4:-0}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${RUN_ROOT}/00_preflight" "$(dirname "${C4_CALIBRATION_PATH}")"

if [[ "${GISP_BUILD_C4}" == "1" ]]; then
  needs_c4=1
  if [[ -f "${C4_CALIBRATION_PATH}" ]]; then
    line_count="$(wc -l < "${C4_CALIBRATION_PATH}" | tr -d ' ')"
    if [[ "${line_count}" == "${C4_CALIBRATION_SAMPLES}" ]]; then
      needs_c4=0
      echo "SKIP C4 calibration artifact; existing ${C4_CALIBRATION_PATH} has ${line_count} rows."
    else
      echo "Regenerating C4 calibration artifact; ${C4_CALIBRATION_PATH} has ${line_count} rows, expected ${C4_CALIBRATION_SAMPLES}."
    fi
  fi

  if [[ "${needs_c4}" == "1" ]]; then
    echo "START build C4 calibration artifact: ${C4_CALIBRATION_PATH}"
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
    "${PYTHON_BIN}" -m src.data.build_c4_calibration \
      --output "${C4_CALIBRATION_PATH}" \
      --samples "${C4_CALIBRATION_SAMPLES}" \
      --seed "${SEED}" \
      --min-chars "${C4_CALIBRATION_MIN_CHARS}" \
      --buffer-size "${C4_CALIBRATION_BUFFER_SIZE}"
    echo "DONE build C4 calibration artifact"
  fi
else
  echo "SKIP C4 calibration artifact; GISP_BUILD_C4=${GISP_BUILD_C4}"
fi

echo "START preflight for ${RUN_ROOT}"
HF_ENDPOINT="${HF_ENDPOINT}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage preflight \
  --force
echo "DONE preflight"

prepare_artifact() {
  local method="$1"
  local artifact="$2"
  local log_path="${LOG_DIR}/${method}_prepare_artifact.log"
  if [[ ",${FINAL_METHODS}," != *",${method},"* ]]; then
    echo "SKIP GISP artifact for ${method}; not requested by STAGE_FINAL_METHODS=${FINAL_METHODS}"
    return
  fi
  if [[ "${FORCE_GISP_PRECOMPUTE:-0}" == "1" || ! -f "${artifact}" ]]; then
    echo "START GISP artifact for ${method}: ${artifact}"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
    "${PYTHON_BIN}" -m src.baselines.prepare_gisp_mlp_artifact \
      --config "${CONFIG_PATH}" \
      --method "${method}" \
      --output "${artifact}" \
      > "${log_path}" 2>&1
    echo "DONE GISP artifact for ${method}"
  else
    echo "SKIP GISP artifact for ${method}; existing ${artifact}"
  fi
}

prepare_artifact "gisp_mlp_t30" "${RUN_ROOT}/00_preflight/gisp_mlp_t30_artifact.json"
prepare_artifact "gisp_mlp_t35" "${RUN_ROOT}/00_preflight/gisp_mlp_t35_artifact.json"
prepare_artifact "gisp_mlp_t30_mixed" "${RUN_ROOT}/00_preflight/gisp_mlp_t30_mixed_artifact.json"
prepare_artifact "gisp_mlp_t20p22_mixed" "${RUN_ROOT}/00_preflight/gisp_mlp_t20p22_mixed_artifact.json"

echo "START ${RUN_LABEL}; methods=${FINAL_METHODS}; shards=${SHARD_COUNT}; global_limit=${FINAL_EVAL_LIMIT}"
pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  shard_summary="$(printf "%s/06_final/summary_shard_%05d_of_%05d.json" "${RUN_ROOT}" "${shard_index}" "${SHARD_COUNT}")"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${shard_summary}" ]]; then
    echo "SKIP shard ${shard_index}/${SHARD_COUNT}; existing ${shard_summary}"
    continue
  fi
  log_path="${LOG_DIR}/${RUN_LABEL}_seed${SEED}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
  echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_SEED="${SEED}" \
  STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
  STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
  STAGE_FINAL_METHODS="${FINAL_METHODS}" \
  STAGE_FINAL_SHARD_INDEX="${shard_index}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${PROFILE}" \
    --stage evaluate_final \
    --force \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one ${RUN_LABEL} shard failed. Check logs under ${LOG_DIR}." >&2
  exit 1
fi

echo "START merge_final_shards"
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage merge_final_shards \
  --force
echo "ALL DONE: ${RUN_ROOT}/06_final/summary.json"
