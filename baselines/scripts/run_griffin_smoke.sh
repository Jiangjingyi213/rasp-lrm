#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/run_external_repo_smoke.sh" "griffin" "external_repos/GRIFFIN" "${1:-Qwen/Qwen3-1.7B}"
