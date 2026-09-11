#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --job-name=rasp_stage_smoke
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
export PYTHON="${PYTHON:-${RASP_ENV_PREFIX}/bin/python}"
export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3.yaml}"
export PROFILE="${PROFILE:-smoke}"
export STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR}/pycache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p logs/slurm "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "config=${CONFIG}"
echo "profile=${PROFILE}"
echo "dtype=${STAGE_MODEL_DTYPE}"
echo "offline_hf=${HF_HUB_OFFLINE}"
nvidia-smi

bash scripts/run_stage_calibrated_pruning.sh
