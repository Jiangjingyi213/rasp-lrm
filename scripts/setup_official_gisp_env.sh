#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GISP_REPO_DIR="${GISP_REPO_DIR:-external_repos/GISP}"
LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/04_gisp_mlp/official_gisp_qwen3_8b}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
INSTALL_GISP_REQUIREMENTS="${INSTALL_GISP_REQUIREMENTS:-0}"
GISP_DEEPSPEED_PACKAGE="${GISP_DEEPSPEED_PACKAGE:-deepspeed==0.14.4}"
GISP_MINIMAL_PACKAGES="${GISP_MINIMAL_PACKAGES:-fire ${GISP_DEEPSPEED_PACKAGE} setuptools jsonlines omegaconf hydra-core einops}"

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
if [[ "${INSTALL_GISP_REQUIREMENTS}" == "1" ]]; then
  mapfile -t requirement_files < <(find "${GISP_REPO_DIR}" -maxdepth 4 -type f \( -iname 'requirements.txt' -o -iname 'requirements*.txt' \) | sort)
  if [[ "${#requirement_files[@]}" -gt 0 ]]; then
    for req in "${requirement_files[@]}"; do
      req_dir="$(dirname "${req}")"
      req_file="$(basename "${req}")"
      echo "pip install -r ${req} from cwd=${req_dir}"
      (
        cd "${req_dir}"
        "${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} -r "${req_file}"
      )
    done
  else
    echo "No requirements.txt found under ${GISP_REPO_DIR}."
  fi
else
  echo "SKIP official requirements; INSTALL_GISP_REQUIREMENTS=${INSTALL_GISP_REQUIREMENTS}"
  echo "Using the existing environment and installing only minimal GISP entrypoint packages."
fi

echo "START install minimal official-GISP runtime dependencies"
DS_BUILD_OPS="${DS_BUILD_OPS:-0}" \
"${PYTHON_BIN}" -m pip install ${PIP_EXTRA_ARGS} ${GISP_MINIMAL_PACKAGES}

echo "START official GISP import smoke check"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

checks = [
    ("fire", "fire"),
    ("deepspeed", "deepspeed"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("jsonlines", "jsonlines"),
    ("omegaconf", "omegaconf"),
    ("hydra", "hydra-core"),
    ("einops", "einops"),
]
optional_checks = [
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("yaml", "pyyaml"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
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
optional_missing = []
for module_name, package_name in optional_checks:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        optional_missing.append((module_name, package_name, repr(exc)))
if optional_missing:
    print("Optional imports missing; only install them if official GISP later requires them:")
    for module_name, package_name, exc in optional_missing:
        print(f"  {module_name} ({package_name}): {exc}")
print("official GISP dependency smoke check passed")
PY

echo "DONE setup official GISP env"
