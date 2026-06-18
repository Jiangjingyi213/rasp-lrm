#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"

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
    raise SystemExit("CUDA is required for the stage-calibrated pruning workflow")
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

PYTHONPYCACHEPREFIX=/tmp/rasp-stage-calibration-pycache \
"${PYTHON}" -m unittest \
  tests.test_stage_calibration_protocol \
  tests.test_stage_calibration_pool \
  tests.test_stage_calibration_statistics \
  tests.test_stage_calibration_runtime \
  tests.test_answer_match \
  tests.test_format_prompt
