#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p external_repos external_outputs runs/02_baselines/external_baselines

clone_or_update() {
  local name="$1"
  local repo="$2"
  local dir="$3"

  if [[ -d "$dir/.git" ]]; then
    echo "[skip] $name already exists at $dir"
  else
    echo "[clone] $name -> $dir"
    git clone --depth 1 "$repo" "$dir"
  fi

  echo "[commit] $name"
  git -C "$dir" rev-parse HEAD
}

clone_or_update "FLAP" "https://github.com/CASIA-LMC-Lab/FLAP.git" "external_repos/FLAP"
clone_or_update "LLM-Pruner" "https://github.com/horseee/LLM-Pruner.git" "external_repos/LLM-Pruner"
clone_or_update "GRIFFIN" "https://github.com/hdong920/GRIFFIN.git" "external_repos/GRIFFIN"

echo "External baseline repositories are ready under external_repos/."
