#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --job-name=rasp_stage_preflight
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
PYTHON="${PYTHON:-${RASP_ENV_PREFIX}/bin/python}"
CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3.yaml}"
PROFILE="${PROFILE:-smoke}"
export STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR}/pycache}"

mkdir -p logs/slurm "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "config=${CONFIG}"
echo "profile=${PROFILE}"
echo "dtype=${STAGE_MODEL_DTYPE}"
nvidia-smi

"${PYTHON}" - <<'PY'
import sys

import torch
import transformers
from math_verify import parse, verify

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 required, found {sys.version.split()[0]}")
if not verify(parse(r"$\boxed{\frac{1}{2}}$"), parse("0.5")):
    raise SystemExit("Math-Verify smoke failed")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("cuda_device_name", torch.cuda.get_device_name(0))
PY

"${PYTHON}" -m unittest \
  tests.test_stage_calibration_protocol \
  tests.test_stage_calibration_pool \
  tests.test_stage_calibration_statistics \
  tests.test_stage_calibration_runtime \
  tests.test_answer_match \
  tests.test_format_prompt

"${PYTHON}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG}" \
  --profile "${PROFILE}" \
  --stage preflight \
  --force
