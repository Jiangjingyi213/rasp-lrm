#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-scx9ftv@BSCC-N32-H@ssh.paracloud.com}"
PORT="${PORT:-2222}"
REMOTE_DIR="${REMOTE_DIR:-/home/bingxing2/home/scx9ftv/workspace/projects/rasp-lrm}"

echo "Creating remote directory: ${REMOTE_DIR}"
ssh -p "${PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

echo "Syncing ${ROOT} -> ${REMOTE}:${REMOTE_DIR}"
rsync -az --progress \
  -e "ssh -p ${PORT}" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".codex-python-packages/" \
  --exclude ".cache/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude "runs/*" \
  --exclude "outputs/" \
  --exclude "data/raw/" \
  --exclude "data/processed/" \
  --exclude "data/trajectories/" \
  --exclude "external_repos/" \
  --exclude "external_outputs/" \
  --exclude "external_envs/" \
  --exclude "checkpoints/" \
  --exclude "huggingface/" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "*.bin" \
  --exclude "*.safetensors" \
  "${ROOT}/" \
  "${REMOTE}:${REMOTE_DIR}/"

echo "Sync complete."
