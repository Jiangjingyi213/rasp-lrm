#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"

for dataset in gsm8k math500; do
  input="runs/02_baselines/eval_dense_qwen3_${dataset}_budget/01_trajectories.jsonl"
  output="runs/02_baselines/eval_dense_qwen3_${dataset}_budget/01_trajectories_math_verify.jsonl"
  summary="runs/02_baselines/eval_dense_qwen3_${dataset}_budget/01_trajectories_math_verify.summary.json"
  if [[ ! -f "${input}" ]]; then
    echo "Missing historical baseline: ${input}" >&2
    exit 1
  fi
  "${PYTHON}" -m src.main_rescore_trajectories \
    --input "${input}" \
    --output "${output}" \
    --summary "${summary}"
done
