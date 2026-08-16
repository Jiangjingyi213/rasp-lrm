#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CONFIG="${CONFIG:-configs/generated_multi_structure_stage_budget_v1/attention_vs_mlp_1040_same_pruning.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/06_attention_vs_mlp_h06_600}"
export LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/06_attention_vs_mlp_h06_600}"
export STAGE_ATTENTION_MLP_GSM8K_LIMIT="${STAGE_ATTENTION_MLP_GSM8K_LIMIT:-300}"
export STAGE_ATTENTION_MLP_MATH500_LIMIT="${STAGE_ATTENTION_MLP_MATH500_LIMIT:-300}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-8}"
export SUITE_SCHEMA="${SUITE_SCHEMA:-attention_vs_mlp_h06_600_aggregate_v1}"
export SUITE_TITLE="${SUITE_TITLE:-Attention-vs-MLP h06 600 Same-Pruning Aggregate Summary}"
export SUITE_DATASET_NOTE="${SUITE_DATASET_NOTE:-GSM8K 300 + MATH500 300}"
export LOG_PREFIX="${LOG_PREFIX:-attention_vs_mlp_h06_600}"

echo "Launching ${SUITE_TITLE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "FINAL_GPUS=${FINAL_GPUS}"
echo "STAGE_FINAL_SHARD_COUNT=${STAGE_FINAL_SHARD_COUNT}"
echo "Limits: gsm8k=${STAGE_ATTENTION_MLP_GSM8K_LIMIT}, math500=${STAGE_ATTENTION_MLP_MATH500_LIMIT}"

bash scripts/run_attention_vs_mlp_h06_1040_8gpu.sh
