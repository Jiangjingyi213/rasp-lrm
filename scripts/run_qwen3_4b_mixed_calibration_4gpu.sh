#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_qwen3_4b_seed3.yaml}"
export PROFILE="${PROFILE:-pilot}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export LOG_DIR="${LOG_DIR:-logs}"

exec bash scripts/run_stage_calibrated_pruning_4gpu.sh
