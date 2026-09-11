#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=rasp_gpu_smoke
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
PYTHON="${PYTHON:-${RASP_ENV_PREFIX}/bin/python}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"

mkdir -p logs/slurm manifests "${HF_HOME}" "${TMPDIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "workdir=$(pwd)"
echo "python=${PYTHON}"
uname -m
nvidia-smi

"${PYTHON}" - <<'PY'
import platform
import sys

import torch
import transformers

print("python", sys.version.split()[0])
print("arch", platform.machine())
print("torch", torch.__version__)
print("torch_file", torch.__file__)
print("torch_cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul_sum", float(y.sum().detach().cpu()))
print("transformers", transformers.__version__)
PY
