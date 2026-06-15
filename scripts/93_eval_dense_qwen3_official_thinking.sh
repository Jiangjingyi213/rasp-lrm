#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

if [[ "${#GPUS[@]}" -lt 2 ]]; then
  echo "Provide two GPU IDs, for example GPU_IDS=0,1" >&2
  exit 1
fi

mkdir -p logs/02_baselines/official_thinking

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -m src.main_generate \
  --config configs/eval_dense_qwen3_gsm8k_official_thinking.yaml \
  > logs/02_baselines/official_thinking/gsm8k.log 2>&1 &
gsm_pid=$!

CUDA_VISIBLE_DEVICES="${GPUS[1]}" "${PYTHON}" -m src.main_generate \
  --config configs/eval_dense_qwen3_math500_official_thinking.yaml \
  > logs/02_baselines/official_thinking/math500.log 2>&1 &
math_pid=$!

status=0
wait "${gsm_pid}" || status=1
wait "${math_pid}" || status=1
exit "${status}"
