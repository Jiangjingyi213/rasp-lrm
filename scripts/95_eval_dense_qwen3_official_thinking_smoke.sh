#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GPU_ID="${GPU_ID:-0}"

"${PYTHON}" scripts/94_prepare_official_thinking_smoke_configs.py

for dataset in gsm8k math500; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m src.main_generate \
    --config "configs/generated_official_thinking_smoke/${dataset}.yaml"
done
