#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GISP_REPO_DIR="${GISP_REPO_DIR:-external_repos/GISP}"
LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/04_gisp_mlp/official_gisp_qwen3_8b}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.version)
PY

echo "START install official GISP requirements"
mapfile -t requirement_files < <(find "${GISP_REPO_DIR}" -maxdepth 4 -type f \( -iname 'requirements.txt' -o -iname 'requirements*.txt' \) | sort)
if [[ "${#requirement_files[@]}" -gt 0 ]]; then
  for req in "${requirement_files[@]}"; do
    echo "pip install -r ${req}"
    "${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} -r "${req}"
  done
else
  echo "No requirements.txt found under ${GISP_REPO_DIR}; installing known official-GISP runtime dependencies."
fi

echo "START install known official-GISP runtime dependencies"
DS_BUILD_OPS="${DS_BUILD_OPS:-0}" \
"${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} \
  fire \
  deepspeed \
  accelerate \
  datasets \
  sentencepiece \
  protobuf \
  jsonlines \
  omegaconf \
  hydra-core \
  einops

echo "START official GISP import smoke check"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

checks = [
    ("fire", "fire"),
    ("deepspeed", "deepspeed"),
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("yaml", "pyyaml"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
    ("jsonlines", "jsonlines"),
    ("omegaconf", "omegaconf"),
    ("hydra", "hydra-core"),
    ("einops", "einops"),
]
missing = []
for module_name, package_name in checks:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append((module_name, package_name, repr(exc)))
if missing:
    print("Missing or broken imports:", file=sys.stderr)
    for module_name, package_name, exc in missing:
        print(f"  {module_name} ({package_name}): {exc}", file=sys.stderr)
    sys.exit(3)
print("official GISP dependency smoke check passed")
PY

echo "DONE setup official GISP env"
