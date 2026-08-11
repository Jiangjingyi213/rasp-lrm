#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/11_multi_structure_stage_budget_v1}"
LOG_ROOT="${LOG_ROOT:-logs/11_multi_structure_stage_budget_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
SEED="${STAGE_SEED:-3}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-4}"
export STAGE_FINAL_SEEDS="${STAGE_FINAL_SEEDS:-${SEED}}"

mkdir -p \
  "${RUN_ROOT}/00_preflight" \
  "${RUN_ROOT}/01_smoke" \
  "${RUN_ROOT}/02_attention_only_math1000" \
  "${RUN_ROOT}/03_joint_allocation_math1000" \
  "${RUN_ROOT}/04_analysis" \
  "${LOG_ROOT}/01_smoke" \
  "${LOG_ROOT}/02_attention_only_math1000" \
  "${LOG_ROOT}/03_joint_allocation_math1000"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to the Qwen3-1.7B pilot run with 03_selected and 04_masks." >&2
    exit 2
  fi
done

write_enriched_aggregate() {
  local suite_root="$1"
  "${PYTHON_BIN}" - "${suite_root}" <<'PY'
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
items = []

def pct(value):
    if value is None:
        return "-"
    return f"{100 * float(value):.2f}%"

for summary_path in sorted(suite.glob("*/06_final/summary.json")):
    data = json.load(open(summary_path, "r", encoding="utf-8"))
    run_root = summary_path.parents[1]
    for dataset_name, summaries in data.get("datasets", {}).items():
        rows = []
        for summary in sorted(summaries, key=lambda row: row["method"]["name"]):
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": summary["method"]["name"],
                    "problems": summary.get("problems", 0),
                    "correct": summary.get("correct", 0),
                    "accuracy": summary.get("accuracy"),
                    "mlp_pruning": summary.get(
                        "actual_average_mlp_pruning_ratio",
                        summary.get("theoretical_average_mlp_pruning_ratio"),
                    ),
                    "attention_pruning": summary.get(
                        "actual_average_attention_head_pruning_ratio"
                    ),
                    "fallback": summary.get("fallback_rate"),
                    "truncation": summary.get("truncation_rate"),
                    "mean_tokens": summary.get("mean_generated_tokens"),
                }
            )
        items.append({"dataset": dataset_name, "run_root": str(run_root), "rows": rows})

aggregate = {
    "schema": "multi_structure_stage_budget_aggregate_v1",
    "completed_datasets": [row["dataset"] for row in items],
    "datasets": items,
}
(suite / "aggregate_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# Multi-Structure Stage-Budget Aggregate Summary",
    "",
    f"- completed_datasets: {len(items)}",
    "",
    "| dataset | method | correct / total | accuracy | MLP pruning | attention head pruning | fallback | truncation | mean tokens |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for item in items:
    for row in item["rows"]:
        mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
        lines.append(
            f"| `{item['dataset']}` | `{row['method']}` | "
            f"{int(row['correct'])} / {int(row['problems'])} | {pct(row['accuracy'])} | "
            f"{pct(row['mlp_pruning'])} | {pct(row['attention_pruning'])} | "
            f"{pct(row['fallback'])} | {pct(row['truncation'])} | {mean_tokens} |"
        )
(suite / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_dataset() {
  local config="$1"
  local suite_root="$2"
  local log_dir="$3"
  local methods="$4"
  local dataset="$5"
  local limit="$6"
  local prefix="$7"
  CONFIG="${config}" \
  RUN_ROOT="${suite_root}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  LOG_DIR="${log_dir}" \
  LOG_PREFIX="${prefix}" \
  STAGE_SEED="${SEED}" \
  STAGE_FINAL_SEEDS="${SEED}" \
  DATASETS_OVERRIDE="${dataset}" \
  STAGE_FINAL_EVAL_LIMIT="${limit}" \
  STAGE_FINAL_METHODS="${methods}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
}

run_phase() {
  local phase_name="$1"
  local config="$2"
  local suite_root="$3"
  local log_dir="$4"
  local methods="$5"
  local default_limit="$6"
  shift 6
  echo "==== START ${phase_name}; methods=${methods} ===="
  while [[ "$#" -gt 0 ]]; do
    local dataset="$1"
    local limit="$2"
    shift 2
    if [[ "${default_limit}" != "-" ]]; then
      limit="${default_limit}"
    fi
    run_dataset "${config}" "${suite_root}" "${log_dir}" "${methods}" "${dataset}" "${limit}" "${phase_name}"
    write_enriched_aggregate "${suite_root}"
  done
  echo "==== DONE ${phase_name}; summary=${suite_root}/aggregate_summary.md ===="
}

SMOKE_CONFIG="${SMOKE_CONFIG:-configs/generated_multi_structure_stage_budget_v1/smoke.yaml}"
ATTN_CONFIG="${ATTN_CONFIG:-configs/generated_multi_structure_stage_budget_v1/math1000_attention_only.yaml}"
JOINT_CONFIG="${JOINT_CONFIG:-configs/generated_multi_structure_stage_budget_v1/math1000_joint_allocation.yaml}"

SMOKE_METHODS="mlp_only_v41_plus,attention_head_only_stage_budget_h10,multi_structure_stage_budget_v1"
ATTN_METHODS="structured_dense,attention_head_only_fixed_h06,attention_head_only_fixed_h10,attention_head_only_stage_budget_h10"
JOINT_METHODS="mlp_only_v41_plus,mlp_v41_plus_attention_fixed_h06,multi_structure_stage_budget_v1"

run_phase \
  "smoke" \
  "${SMOKE_CONFIG}" \
  "${RUN_ROOT}/01_smoke" \
  "${LOG_ROOT}/01_smoke" \
  "${SMOKE_METHODS}" \
  "8" \
  gsm8k 8 \
  math500 8 \
  amc2023 8 \
  aime2024 8 \
  aime2025 8 \
  gpqa_diamond 8

run_phase \
  "attention_only_math1000" \
  "${ATTN_CONFIG}" \
  "${RUN_ROOT}/02_attention_only_math1000" \
  "${LOG_ROOT}/02_attention_only_math1000" \
  "${ATTN_METHODS}" \
  "-" \
  gsm8k 300 \
  math500 400 \
  amc2023 40 \
  aime2024 30 \
  aime2025 30 \
  gpqa_diamond 198

run_phase \
  "joint_allocation_math1000" \
  "${JOINT_CONFIG}" \
  "${RUN_ROOT}/03_joint_allocation_math1000" \
  "${LOG_ROOT}/03_joint_allocation_math1000" \
  "${JOINT_METHODS}" \
  "-" \
  gsm8k 300 \
  math500 400 \
  amc2023 40 \
  aime2024 30 \
  aime2025 30 \
  gpqa_diamond 198

echo "ALL DONE: ${RUN_ROOT}"
