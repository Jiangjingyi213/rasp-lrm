#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 METHOD_NAME REPO_DIR [MODEL_NAME]" >&2
  exit 2
fi

METHOD="$1"
REPO_DIR="$2"
MODEL_NAME="${3:-Qwen/Qwen3-1.7B}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/runs/02_baselines/external_baselines"
OUT="$OUT_DIR/${METHOD}_smoke.txt"

mkdir -p "$OUT_DIR"

{
  echo "method: $METHOD"
  echo "repo_dir: $REPO_DIR"
  echo "model_name: $MODEL_NAME"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  if [[ ! -d "$ROOT/$REPO_DIR/.git" ]]; then
    echo "status: missing_repo"
    echo "Run: bash baselines/scripts/clone_external_baselines.sh"
    exit 0
  fi

  cd "$ROOT/$REPO_DIR"
  echo "status: repo_found"
  echo "commit: $(git rev-parse HEAD)"
  echo

  echo "requirements:"
  find . -maxdepth 2 \( -iname "requirements*.txt" -o -iname "environment*.yml" -o -iname "pyproject.toml" -o -iname "setup.py" \) | sort
  echo

  echo "model_support_string_hits:"
  grep -RInE "Qwen|qwen|Llama|llama|Vicuna|BLOOM|Baichuan|TinyLlama" . \
    --exclude-dir=.git --exclude='*.ipynb' 2>/dev/null | head -n 80 || true
  echo

  echo "likely_entrypoints:"
  find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" \) | sort | head -n 120
  echo

  echo "readme_head:"
  if [[ -f README.md ]]; then
    sed -n '1,160p' README.md
  elif [[ -f readme.md ]]; then
    sed -n '1,160p' readme.md
  else
    echo "No README found."
  fi
} > "$OUT"

echo "Wrote smoke report: $OUT"
