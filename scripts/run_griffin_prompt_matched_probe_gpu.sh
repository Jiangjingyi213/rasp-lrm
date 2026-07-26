#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/griffin_prompt_matched_probe.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_griffin_prompt_matched_probe}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-griffin_t20_nominal_0p24,griffin_t20_nominal_0p27,griffin_t30_nominal_0p37,griffin_t30_nominal_0p40}"
export STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:-100}"
export RUN_LABEL="${RUN_LABEL:-griffin_prompt_matched_probe}"

exec bash scripts/run_griffin_prompt_matched_full_gpu.sh
