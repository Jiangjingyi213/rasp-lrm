#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GISP_REPO_URL="${GISP_REPO_URL:-https://github.com/uncc-efficient-ai/GISP.git}"
GISP_REPO_DIR="${GISP_REPO_DIR:-external_repos/GISP}"
GISP_CLONE_RETRIES="${GISP_CLONE_RETRIES:-3}"
GISP_ARCHIVE_PATH="${GISP_ARCHIVE_PATH:-}"
GISP_TARBALL_URL="${GISP_TARBALL_URL:-}"
RUN_ROOT="${RUN_ROOT:-runs/12_additional_baselines/04_gisp_mlp/05_official_gisp_qwen3_8b_c4_t20_gsm8k_full}"
LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/04_gisp_mlp/official_gisp_qwen3_8b}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/generated_additional_baselines/official_gisp_qwen3_8b_gsm8k_clean_eval.yaml}"
PROFILE="${PROFILE:-pilot}"
SEED="${STAGE_SEED:-3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
PRUNING_RATIO="${PRUNING_RATIO:-0.20}"
GISP_ITERATIONS="${GISP_ITERATIONS:-112}"
GISP_SEQ_LEN="${GISP_SEQ_LEN:-256}"
GISP_CALIBRATION_SAMPLES="${GISP_CALIBRATION_SAMPLES:-2000}"
GISP_PRUNE_BATCH_SIZE="${GISP_PRUNE_BATCH_SIZE:-1}"
GISP_CALIBRATION_MIN_CHARS="${GISP_CALIBRATION_MIN_CHARS:-64}"
GISP_CALIBRATION_BUFFER_SIZE="${GISP_CALIBRATION_BUFFER_SIZE:-10000}"

C4_CALIBRATION_PATH="${C4_CALIBRATION_PATH:-${RUN_ROOT}/00_c4_calibration/c4_${GISP_CALIBRATION_SAMPLES}_seed${SEED}.jsonl}"
OFFICIAL_CONFIG_PATH="${OFFICIAL_CONFIG_PATH:-${RUN_ROOT}/00_official_gisp/gisp_qwen3_8b_c4_t20.yaml}"
OFFICIAL_CONFIG_MANIFEST="${OFFICIAL_CONFIG_MANIFEST:-${RUN_ROOT}/00_official_gisp/gisp_qwen3_8b_c4_t20.config_manifest.json}"
PRUNED_MODEL_DIR="${PRUNED_MODEL_DIR:-${RUN_ROOT}/01_pruned_model}"

RUN_LABEL="${RUN_LABEL:-official_gisp_qwen3_8b_c4_t20_gsm8k_full}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_GISP_PRUNE="${SKIP_GISP_PRUNE:-0}"
SETUP_GISP_ENV="${SETUP_GISP_ENV:-1}"
INSTALL_GISP_REQUIREMENTS="${INSTALL_GISP_REQUIREMENTS:-0}"
GISP_CONFIG_ARG="${GISP_CONFIG_ARG:---config_path}"
GISP_PRUNE_GPUS="${GISP_PRUNE_GPUS:-0,1,2,3,4}"
PATCH_GISP_QWEN3_LOADER="${PATCH_GISP_QWEN3_LOADER:-1}"
CHECK_QWEN3_AUTO_CLASS="${CHECK_QWEN3_AUTO_CLASS:-1}"
GISP_TORCHRUN_ARGS="${GISP_TORCHRUN_ARGS:---standalone --nnodes=1}"

IFS=',' read -r -a GISP_PRUNE_GPU_ARRAY <<< "${GISP_PRUNE_GPUS}"
GISP_PRUNE_GPU_COUNT="${#GISP_PRUNE_GPU_ARRAY[@]}"
if [[ -z "${GISP_ENABLE_PIPELINE:-}" ]]; then
  if [[ "${GISP_PRUNE_GPU_COUNT}" -gt 1 ]]; then
    GISP_ENABLE_PIPELINE=1
  else
    GISP_ENABLE_PIPELINE=0
  fi
fi
if [[ -z "${GISP_PIPELINE_NODES:-}" ]]; then
  GISP_PIPELINE_NODES="$(seq -s, 0 $((GISP_PRUNE_GPU_COUNT - 1)))"
fi
GISP_PIPELINE_ARGS=()
if [[ "${GISP_ENABLE_PIPELINE}" == "1" ]]; then
  GISP_PIPELINE_ARGS+=(--enable-pipeline)
fi

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "$(dirname "${C4_CALIBRATION_PATH}")" "$(dirname "${OFFICIAL_CONFIG_PATH}")"

if [[ ! -d "${GISP_REPO_DIR}/.git" ]]; then
  if [[ -n "${GISP_ARCHIVE_PATH}" ]]; then
    echo "START extract official GISP archive: ${GISP_ARCHIVE_PATH} -> ${GISP_REPO_DIR}"
    tmp_extract="${RUN_ROOT}/00_official_gisp/gisp_archive_extract"
    rm -rf "${tmp_extract}"
    mkdir -p "${tmp_extract}" "$(dirname "${GISP_REPO_DIR}")"
    tar -xf "${GISP_ARCHIVE_PATH}" -C "${tmp_extract}"
    extracted_root="$(find "${tmp_extract}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -z "${extracted_root}" ]]; then
      echo "GISP_ARCHIVE_PATH did not contain a top-level directory: ${GISP_ARCHIVE_PATH}" >&2
      exit 7
    fi
    rm -rf "${GISP_REPO_DIR}"
    mv "${extracted_root}" "${GISP_REPO_DIR}"
    echo "DONE extract official GISP archive"
  elif [[ -n "${GISP_TARBALL_URL}" ]]; then
    echo "START download official GISP tarball: ${GISP_TARBALL_URL}"
    tmp_tar="${RUN_ROOT}/00_official_gisp/gisp_source.tar.gz"
    curl -L --connect-timeout 30 --max-time 300 -o "${tmp_tar}" "${GISP_TARBALL_URL}"
    tmp_extract="${RUN_ROOT}/00_official_gisp/gisp_tarball_extract"
    rm -rf "${tmp_extract}"
    mkdir -p "${tmp_extract}" "$(dirname "${GISP_REPO_DIR}")"
    tar -xf "${tmp_tar}" -C "${tmp_extract}"
    extracted_root="$(find "${tmp_extract}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -z "${extracted_root}" ]]; then
      echo "GISP_TARBALL_URL did not contain a top-level directory: ${GISP_TARBALL_URL}" >&2
      exit 8
    fi
    rm -rf "${GISP_REPO_DIR}"
    mv "${extracted_root}" "${GISP_REPO_DIR}"
    echo "DONE download official GISP tarball"
  else
    echo "START clone official GISP: ${GISP_REPO_URL} -> ${GISP_REPO_DIR}"
    cloned=0
    for attempt in $(seq 1 "${GISP_CLONE_RETRIES}"); do
      echo "Clone attempt ${attempt}/${GISP_CLONE_RETRIES}"
      rm -rf "${GISP_REPO_DIR}"
      if git -c http.version=HTTP/1.1 clone --depth 1 "${GISP_REPO_URL}" "${GISP_REPO_DIR}"; then
        cloned=1
        break
      fi
      sleep 5
    done
    if [[ "${cloned}" != "1" ]]; then
      echo "Failed to clone official GISP after ${GISP_CLONE_RETRIES} attempts." >&2
      echo "Retry with GISP_TARBALL_URL=https://github.com/uncc-efficient-ai/GISP/archive/refs/heads/main.tar.gz" >&2
      echo "or place a downloaded archive on the server and set GISP_ARCHIVE_PATH=/path/to/GISP-main.tar.gz." >&2
      exit 9
    fi
    echo "DONE clone official GISP"
  fi
else
  echo "SKIP clone official GISP; existing ${GISP_REPO_DIR}"
fi

if [[ "${SETUP_GISP_ENV}" == "1" ]]; then
  echo "START setup official GISP env"
  PYTHON="${PYTHON_BIN}" \
  GISP_REPO_DIR="${GISP_REPO_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  INSTALL_GISP_REQUIREMENTS="${INSTALL_GISP_REQUIREMENTS}" \
  bash scripts/setup_official_gisp_env.sh > "${LOG_DIR}/${RUN_LABEL}_setup_env.log" 2>&1
  echo "DONE setup official GISP env"
else
  echo "SKIP setup official GISP env; SETUP_GISP_ENV=${SETUP_GISP_ENV}"
fi

if [[ "${PATCH_GISP_QWEN3_LOADER}" == "1" ]]; then
  echo "START patch official GISP Qwen3 HF loader"
  "${PYTHON_BIN}" tools/official_gisp/patch_qwen3_hf_loader.py \
    --gisp-repo-dir "${GISP_REPO_DIR}"
  echo "DONE patch official GISP Qwen3 HF loader"
else
  echo "SKIP patch official GISP Qwen3 HF loader; PATCH_GISP_QWEN3_LOADER=${PATCH_GISP_QWEN3_LOADER}"
fi
if grep -RIn "Qwen2ForCausalLM" "${GISP_REPO_DIR}" --include="*.py"; then
  echo "Qwen2ForCausalLM remains in official GISP Python files after patch; aborting." >&2
  exit 10
fi
"${PYTHON_BIN}" tools/official_gisp/check_gisp_patch_state.py \
  --gisp-repo-dir "${GISP_REPO_DIR}"

if [[ "${CHECK_QWEN3_AUTO_CLASS}" == "1" ]]; then
  echo "START check Qwen3 AutoModelForCausalLM mapping"
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  "${PYTHON_BIN}" tools/official_gisp/check_qwen3_auto_model.py \
    --model "${BASE_MODEL}"
  echo "DONE check Qwen3 AutoModelForCausalLM mapping"
else
  echo "SKIP check Qwen3 AutoModelForCausalLM mapping; CHECK_QWEN3_AUTO_CLASS=${CHECK_QWEN3_AUTO_CLASS}"
fi

if [[ -f "${C4_CALIBRATION_PATH}" ]]; then
  line_count="$(wc -l < "${C4_CALIBRATION_PATH}" | tr -d ' ')"
  if [[ "${line_count}" == "${GISP_CALIBRATION_SAMPLES}" ]]; then
    echo "SKIP C4 calibration; existing ${C4_CALIBRATION_PATH} has ${line_count} rows."
  else
    echo "C4 calibration row count mismatch: ${line_count}, expected ${GISP_CALIBRATION_SAMPLES}" >&2
    exit 3
  fi
else
  echo "START build C4 calibration: ${C4_CALIBRATION_PATH}"
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  "${PYTHON_BIN}" -m src.data.build_c4_calibration \
    --output "${C4_CALIBRATION_PATH}" \
    --samples "${GISP_CALIBRATION_SAMPLES}" \
    --seed "${SEED}" \
    --min-chars "${GISP_CALIBRATION_MIN_CHARS}" \
    --buffer-size "${GISP_CALIBRATION_BUFFER_SIZE}"
  echo "DONE build C4 calibration"
fi

echo "START make official GISP config: ${OFFICIAL_CONFIG_PATH}"
"${PYTHON_BIN}" tools/official_gisp/make_qwen3_8b_config.py \
  --gisp-repo-dir "${GISP_REPO_DIR}" \
  --output-config "${OFFICIAL_CONFIG_PATH}" \
  --manifest "${OFFICIAL_CONFIG_MANIFEST}" \
  --model "${BASE_MODEL}" \
  --calibration-path "${C4_CALIBRATION_PATH}" \
  --output-model-dir "${PRUNED_MODEL_DIR}" \
  --pruning-ratio "${PRUNING_RATIO}" \
  --iterations "${GISP_ITERATIONS}" \
  --seq-len "${GISP_SEQ_LEN}" \
  --samples "${GISP_CALIBRATION_SAMPLES}" \
  --batch-size "${GISP_PRUNE_BATCH_SIZE}" \
  "${GISP_PIPELINE_ARGS[@]}" \
  --pipeline-nodes "${GISP_PIPELINE_NODES}"
echo "DONE make official GISP config"

if [[ "${SKIP_GISP_PRUNE}" != "1" ]]; then
  if [[ -n "${GISP_PRUNE_CMD:-}" ]]; then
    echo "START official GISP prune via GISP_PRUNE_CMD"
    export GISP_REPO_DIR OFFICIAL_CONFIG_PATH PRUNED_MODEL_DIR BASE_MODEL C4_CALIBRATION_PATH
    export GISP_LOCAL_C4_JSONL="${C4_CALIBRATION_PATH}"
    bash -lc "${GISP_PRUNE_CMD}" > "${LOG_DIR}/${RUN_LABEL}_prune.log" 2>&1
    echo "DONE official GISP prune via GISP_PRUNE_CMD"
  else
    entrypoint=""
    for candidate in \
      "${GISP_REPO_DIR}/external_code/GISP/main.py" \
      "${GISP_REPO_DIR}/main.py" \
      "${GISP_REPO_DIR}/run.py" \
      "${GISP_REPO_DIR}/src/main.py"
    do
      if [[ -f "${candidate}" ]]; then
        entrypoint="${candidate}"
        break
      fi
    done
    if [[ -z "${entrypoint}" ]]; then
      echo "Could not locate an official GISP Python entrypoint." >&2
      echo "Set GISP_PRUNE_CMD with a command that consumes OFFICIAL_CONFIG_PATH." >&2
      find "${GISP_REPO_DIR}" -maxdepth 3 -type f \( -name 'main.py' -o -name 'run.py' -o -name '*.sh' \) | sort >&2
      exit 4
    fi
    if [[ "${GISP_ENABLE_PIPELINE}" == "1" ]]; then
      echo "START official GISP prune with torchrun: nproc=${GISP_PRUNE_GPU_COUNT}; ${entrypoint} ${GISP_CONFIG_ARG} ${OFFICIAL_CONFIG_PATH}"
      CUDA_VISIBLE_DEVICES="${GISP_PRUNE_GPUS}" \
      GISP_LOCAL_C4_JSONL="${C4_CALIBRATION_PATH}" \
      HF_ENDPOINT="${HF_ENDPOINT}" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
      HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      "${PYTHON_BIN}" -m torch.distributed.run ${GISP_TORCHRUN_ARGS} \
        --nproc_per_node "${GISP_PRUNE_GPU_COUNT}" \
        "${entrypoint}" "${GISP_CONFIG_ARG}" "${OFFICIAL_CONFIG_PATH}" \
        > "${LOG_DIR}/${RUN_LABEL}_prune.log" 2>&1
    else
      echo "START official GISP prune: ${entrypoint} ${GISP_CONFIG_ARG} ${OFFICIAL_CONFIG_PATH}"
      CUDA_VISIBLE_DEVICES="${GISP_PRUNE_GPUS}" \
      GISP_LOCAL_C4_JSONL="${C4_CALIBRATION_PATH}" \
      HF_ENDPOINT="${HF_ENDPOINT}" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
      HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
      "${PYTHON_BIN}" "${entrypoint}" "${GISP_CONFIG_ARG}" "${OFFICIAL_CONFIG_PATH}" \
        > "${LOG_DIR}/${RUN_LABEL}_prune.log" 2>&1
    fi
    echo "DONE official GISP prune"
    if grep -E "Qwen2ForCausalLM|model of type qwen3 to instantiate a model of type qwen2" "${LOG_DIR}/${RUN_LABEL}_prune.log"; then
      echo "Official GISP prune log shows Qwen3 was loaded through Qwen2; aborting before evaluation." >&2
      exit 11
    fi
  fi
else
  echo "SKIP official GISP prune; SKIP_GISP_PRUNE=${SKIP_GISP_PRUNE}"
fi

if [[ ! -f "${PRUNED_MODEL_DIR}/config.json" ]]; then
  echo "Missing HuggingFace config.json in PRUNED_MODEL_DIR=${PRUNED_MODEL_DIR}" >&2
  echo "If official GISP saved elsewhere, rerun with PRUNED_MODEL_DIR=/path/to/saved/model and SKIP_GISP_PRUNE=1." >&2
  exit 5
fi

echo "START preflight for evaluation: ${RUN_ROOT}"
HF_ENDPOINT="${HF_ENDPOINT}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
STAGE_MODEL_NAME_OR_PATH="${PRUNED_MODEL_DIR}" \
STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${EVAL_CONFIG}" \
  --profile "${PROFILE}" \
  --stage preflight \
  --force
echo "DONE preflight for evaluation"

echo "START official GISP downstream evaluation; methods=${FINAL_METHODS}; shards=${SHARD_COUNT}"
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
  STAGE_MODEL_NAME_OR_PATH="${PRUNED_MODEL_DIR}" \
  STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}" \
  STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
  STAGE_FINAL_METHODS="${FINAL_METHODS}" \
  STAGE_FINAL_SHARD_INDEX="${shard_index}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${EVAL_CONFIG}" \
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
  echo "At least one official GISP evaluation shard failed. Check logs under ${LOG_DIR}." >&2
  exit 6
fi

echo "START merge official GISP evaluation shards"
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
STAGE_MODEL_NAME_OR_PATH="${PRUNED_MODEL_DIR}" \
STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${EVAL_CONFIG}" \
  --profile "${PROFILE}" \
  --stage merge_final_shards \
  --force
echo "ALL DONE: ${RUN_ROOT}/06_final/summary.json"
