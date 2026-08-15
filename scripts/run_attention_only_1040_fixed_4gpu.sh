#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG="${CONFIG:-configs/generated_multi_structure_stage_budget_v1/attention_only_1040_fixed.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/05_attention_only_1040_fixed}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/05_attention_only_1040_fixed}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
METHODS="${STAGE_FINAL_METHODS:-attention_head_only_stage_budget_h06}"
SEED="${STAGE_SEED:-3}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-4}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to the Qwen3-1.7B pilot run with 03_selected and 04_masks." >&2
    exit 2
  fi
done

write_attention_aggregate() {
  "${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = []

def pct(value):
    if value is None:
        return "-"
    return f"{100 * float(value):.2f}%"

for summary_path in sorted(root.glob("*/06_final/summary.json")):
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
                    "mlp_pruning": summary.get("actual_average_mlp_pruning_ratio"),
                    "attention_pruning": summary.get("actual_average_attention_head_pruning_ratio"),
                    "attention_by_stage": summary.get("actual_attention_head_pruning_ratio_by_stage", {}),
                    "fallback": summary.get("fallback_rate"),
                    "truncation": summary.get("truncation_rate"),
                    "mean_tokens": summary.get("mean_generated_tokens"),
                    "runtime_backend": summary.get("runtime_backend"),
                }
            )
        items.append({"dataset": dataset_name, "run_root": str(run_root), "rows": rows})

aggregate = {
    "schema": "attention_only_1040_fixed_aggregate_v1",
    "reference_note": "Structured dense is intentionally not rerun; compare against the existing 1040 dense reference.",
    "completed_datasets": [row["dataset"] for row in items],
    "datasets": items,
}
(root / "aggregate_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# Attention-Only 1040 Runtime-Fixed Aggregate Summary",
    "",
    "- structured_dense: not rerun; use existing 1040 dense reference for comparison.",
    f"- completed_datasets: {len(items)}",
    "",
    "| dataset | method | correct / total | accuracy | MLP pruning | attention head pruning | fallback | truncation | mean tokens | runtime |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for item in items:
    for row in item["rows"]:
        mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
        lines.append(
            f"| `{item['dataset']}` | `{row['method']}` | "
            f"{int(row['correct'])} / {int(row['problems'])} | {pct(row['accuracy'])} | "
            f"{pct(row['mlp_pruning'])} | {pct(row['attention_pruning'])} | "
            f"{pct(row['fallback'])} | {pct(row['truncation'])} | {mean_tokens} | "
            f"`{row['runtime_backend'] or '-'}` |"
        )
(root / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_dataset() {
  local dataset="$1"
  local limit="$2"
  CONFIG="${CONFIG}" \
  RUN_ROOT="${RUN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  LOG_DIR="${LOG_ROOT}" \
  LOG_PREFIX="attention_only_1040" \
  STAGE_SEED="${SEED}" \
  STAGE_FINAL_SEEDS="${SEED}" \
  DATASETS_OVERRIDE="${dataset}" \
  STAGE_FINAL_EVAL_LIMIT="${limit}" \
  STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
  write_attention_aggregate
}

run_dataset gsm8k "${STAGE_ATTENTION_ONLY_GSM8K_LIMIT:-520}"
run_dataset math500 "${STAGE_ATTENTION_ONLY_MATH500_LIMIT:-520}"

echo "Attention-only 1040 diagnostic complete: ${RUN_ROOT}/aggregate_summary.md"
