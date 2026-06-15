#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing isolated evaluation environment: ${PYTHON}" >&2
  echo "Create it without modifying the historical rasp_qwen3 environment:" >&2
  echo "  conda create -y -p /home/cike/jjy/envs/rasp_qwen3_eval --clone /home/cike/jjy/envs/rasp_qwen3" >&2
  echo "If the cloned environment uses Python < 3.10, create a fresh Python 3.10+ environment instead." >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 is required for Math-Verify; found {sys.version.split()[0]}"
    )
print("python", sys.version.split()[0])
PY

"${PYTHON}" -m pip install \
  "transformers>=4.51.0" \
  "math-verify[antlr4_13_2]>=0.8.0"

"${PYTHON}" - <<'PY'
import torch
import transformers
from math_verify import parse, verify

assert verify(parse(r"$\boxed{\frac{1}{2}}$"), parse("0.5"))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("math_verify_smoke", "ok")
PY
