#!/usr/bin/env bash
set -euo pipefail

export RASP_BANK_GPU_COUNT="${RASP_BANK_GPU_COUNT:-4}"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/30_collect_runtime_bank_formal.sh"
