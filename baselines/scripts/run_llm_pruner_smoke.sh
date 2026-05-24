#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/run_external_repo_smoke.sh" "llm_pruner" "external_repos/LLM-Pruner" "${1:-Qwen/Qwen3-1.7B}"
