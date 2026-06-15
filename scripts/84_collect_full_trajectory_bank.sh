#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROFILE="${PROFILE:-dense_smoke}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export FULL_TRAJECTORY_GPU_COUNT="${#GPUS[@]}"
CONFIG_DIR="configs/generated_full_trajectory_${PROFILE}"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/10_full_trajectory_multi_window/${PROFILE}}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/83_prepare_full_trajectory_bank_configs.py

pids=()
for queue in "${!GPUS[@]}"; do
  gpu="${GPUS[$queue]}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TOKENIZERS_PARALLELISM=false
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      run_dir="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["run_dir"])' "${config}")"
      validation="${run_dir}/07_full_trajectory_bank_validation.json"
      if [[ -f "${validation}" ]] && "${PYTHON}" -c '
import sys
from pathlib import Path
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, read_yaml
v, c = read_json(sys.argv[1]), read_yaml(sys.argv[2])
b = c["aligned_window_bank"]
ok = (
    v.get("status") == "ok"
    and v.get("ratios") == b["ratios"]
    and v.get("boundary_sampling") == "causal_grid"
    and v.get("action_terminal_semantics") == "eos_before_action_window_complete_v1"
    and v.get("configured_decision_start") == b["decision_start"]
    and v.get("configured_decision_stride") == b["decision_stride"]
    and v.get("configured_include_tail_anchor") is True
    and v.get("collection_config_fingerprint") == config_fingerprint(
        c, ("seed", "model", "prompt", "data", "generation", "aligned_window_bank", "stage_sensitivity")
    )
    and all(Path(c["paths"][name]).is_file() for name in ("trajectories", "probe_dataset", "probe_hidden_states"))
)
raise SystemExit(0 if ok else 1)
' "${validation}" "${config}"; then
        echo "SKIP validated shard ${run_dir}"
        continue
      fi
      echo "START ${config}"
      "${PYTHON}" -m src.main_generate --config "${config}"
      "${PYTHON}" -m src.main_collect_aligned_window_bank --config "${config}"
      "${PYTHON}" -m src.main_validate_aligned_window_bank --config "${config}"
      echo "DONE ${config}"
    done < "${CONFIG_DIR}/gpu${queue}.list"
  ) > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "worker pid=$! queue=${queue} gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
