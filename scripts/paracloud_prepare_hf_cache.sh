#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
PYTHON="${PYTHON:-${RASP_ENV_PREFIX}/bin/python}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TMPDIR}" logs/slurm

"${PYTHON}" - <<'PY'
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

model_name = os.environ.get("STAGE_MODEL_NAME_OR_PATH", "Qwen/Qwen3-1.7B")
print(f"Downloading model cache: {model_name}")
snapshot_download(repo_id=model_name, repo_type="model", resume_download=True)

datasets = [
    ("microsoft/orca-math-word-problems-200k", None, "train"),
    ("deepmind/aqua_rat", "raw", "train"),
    ("allenai/openbookqa", "main", "train"),
    ("tau/commonsense_qa", None, "train"),
    ("openai/gsm8k", "main", "test"),
    ("HuggingFaceH4/MATH-500", None, "test"),
]

for name, config, split in datasets:
    args = [name]
    if config:
        args.append(config)
    print(f"Preparing dataset cache: {name} {config or ''} {split}")
    ds = load_dataset(*args, split=split)
    print(f"  rows={len(ds)}")

print("Hugging Face cache is ready.")
PY
