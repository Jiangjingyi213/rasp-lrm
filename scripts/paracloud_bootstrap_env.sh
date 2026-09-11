#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RASP_CACHE_ROOT}/pip}"

mkdir -p "${RASP_ENV_PREFIX%/*}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TMPDIR}" "${PIP_CACHE_DIR}" logs/slurm

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load miniforge3/24.1 || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Try: module load miniforge3/24.1" >&2
  exit 2
fi

if [[ ! -x "${RASP_ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${RASP_ENV_PREFIX}" python=3.10 pip
fi

# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RASP_ENV_PREFIX}"
set -u

python -m pip install --upgrade pip wheel setuptools

echo "Checking PyTorch in ${RASP_ENV_PREFIX}"
if ! python - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
then
  echo "PyTorch is not installed yet. Looking for cluster-provided torch wheels."
  if [[ -n "${TORCH_WHEEL:-}" ]]; then
    echo "Installing PyTorch from TORCH_WHEEL=${TORCH_WHEEL}"
    python -m pip install "${TORCH_WHEEL}"
  else
    mapfile -t torch_wheels < <(
      {
        find /home/bingxing2/apps/package -maxdepth 3 -type f -name 'torch-*.whl' 2>/dev/null
        find /home/bingxing2/apps/package -maxdepth 3 -type f -name 'torchvision-*.whl' 2>/dev/null
        find /home/bingxing2/apps/package -maxdepth 3 -type f -name 'torchaudio-*.whl' 2>/dev/null
      } | sort -u || true
    )
    if [[ "${#torch_wheels[@]}" -eq 1 ]]; then
      echo "Installing PyTorch from ${torch_wheels[0]}"
      python -m pip install "${torch_wheels[0]}"
    else
      echo "PyTorch is not installed in ${RASP_ENV_PREFIX}." >&2
      echo "Set TORCH_WHEEL to the correct aarch64 CUDA torch wheel and rerun, for example:" >&2
      echo "TORCH_WHEEL=/home/bingxing2/apps/package/<torch-wheel>.whl bash scripts/paracloud_bootstrap_env.sh" >&2
      if [[ "${#torch_wheels[@]}" -gt 0 ]]; then
        echo "Candidate wheels:" >&2
        printf '  %s\n' "${torch_wheels[@]}" >&2
      else
        echo "No torch wheel was found under /home/bingxing2/apps/package." >&2
      fi
      exit 2
    fi
  fi
fi

echo "Installing RASP dependencies without replacing PyTorch."
python -m pip install -r requirements-paracloud.txt

python - <<'PY'
import platform
import sys

import torch
import transformers

print("python", sys.version.split()[0])
print("arch", platform.machine())
print("torch", torch.__version__)
print("torch_file", torch.__file__)
print("torch_cuda_runtime", torch.version.cuda)
print("cuda_available_on_this_node", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

echo "Environment ready: ${RASP_ENV_PREFIX}"
