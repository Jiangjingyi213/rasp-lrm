#!/usr/bin/env bash
set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
export RUN_ROOT="${RUN_ROOT:-runs/12_additional_baselines/04_gisp_mlp/06_official_gisp_qwen3_4b_c4_t20_gsm8k_full}"
export LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/04_gisp_mlp/official_gisp_qwen3_4b}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/generated_additional_baselines/official_gisp_qwen3_4b_gsm8k_clean_eval.yaml}"
export OFFICIAL_CONFIG_PATH="${OFFICIAL_CONFIG_PATH:-${RUN_ROOT}/00_official_gisp/gisp_qwen3_4b_c4_t20.yaml}"
export OFFICIAL_CONFIG_MANIFEST="${OFFICIAL_CONFIG_MANIFEST:-${RUN_ROOT}/00_official_gisp/gisp_qwen3_4b_c4_t20.config_manifest.json}"
export RUN_LABEL="${RUN_LABEL:-official_gisp_qwen3_4b_c4_t20_gsm8k_full}"
export GISP_ENABLE_PIPELINE="${GISP_ENABLE_PIPELINE:-}"
export GISP_PRUNE_BATCH_SIZE="${GISP_PRUNE_BATCH_SIZE:-1}"

bash scripts/run_official_gisp_qwen3_8b_gsm8k_clean.sh
