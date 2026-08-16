#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG="${CONFIG:-configs/generated_multi_structure_stage_budget_v1/attention_vs_mlp_1040_same_pruning.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/06_attention_vs_mlp_h06_1040}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1/11_multi_structure_stage_budget_v1/06_attention_vs_mlp_h06_1040}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
METHODS="${STAGE_FINAL_METHODS:-mlp_only_stage_budget_h06,attention_head_only_stage_budget_h06}"
SEED="${STAGE_SEED:-3}"
SUITE_SCHEMA="${SUITE_SCHEMA:-attention_vs_mlp_h06_1040_aggregate_v1}"
SUITE_TITLE="${SUITE_TITLE:-Attention-vs-MLP h06 1040 Same-Pruning Aggregate Summary}"
SUITE_DATASET_NOTE="${SUITE_DATASET_NOTE:-GSM8K 520 + MATH500 520}"
LOG_PREFIX="${LOG_PREFIX:-attention_vs_mlp_h06_1040}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-8}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to the Qwen3-1.7B pilot run with 03_selected and 04_masks." >&2
    exit 2
  fi
done

write_same_rate_aggregate() {
  SUITE_SCHEMA="${SUITE_SCHEMA}" \
  SUITE_TITLE="${SUITE_TITLE}" \
  SUITE_DATASET_NOTE="${SUITE_DATASET_NOTE}" \
  "${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
suite_schema = os.environ["SUITE_SCHEMA"]
suite_title = os.environ["SUITE_TITLE"]
suite_dataset_note = os.environ["SUITE_DATASET_NOTE"]
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
                    "protocol_valid_accuracy": summary.get("protocol_valid_accuracy"),
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

totals = {}
for item in items:
    for row in item["rows"]:
        method = row["method"]
        acc = totals.setdefault(
            method,
            {
                "correct": 0,
                "problems": 0,
                "mlp_pruning_num": 0.0,
                "attention_pruning_num": 0.0,
                "attention_pruning_den": 0,
                "fallback_num": 0.0,
                "truncation_num": 0.0,
                "tokens_num": 0.0,
            },
        )
        n = int(row["problems"])
        acc["correct"] += int(row["correct"])
        acc["problems"] += n
        if row["mlp_pruning"] is not None:
            acc["mlp_pruning_num"] += float(row["mlp_pruning"]) * n
        if row["attention_pruning"] is not None:
            acc["attention_pruning_num"] += float(row["attention_pruning"]) * n
            acc["attention_pruning_den"] += n
        if row["fallback"] is not None:
            acc["fallback_num"] += float(row["fallback"]) * n
        if row["truncation"] is not None:
            acc["truncation_num"] += float(row["truncation"]) * n
        if row["mean_tokens"] is not None:
            acc["tokens_num"] += float(row["mean_tokens"]) * n

weighted = []
for method, row in sorted(totals.items()):
    n = row["problems"]
    weighted.append(
        {
            "method": method,
            "correct": row["correct"],
            "problems": n,
            "accuracy": row["correct"] / n if n else None,
            "mlp_pruning": row["mlp_pruning_num"] / n if n else None,
            "attention_pruning": (
                row["attention_pruning_num"] / row["attention_pruning_den"]
                if row["attention_pruning_den"]
                else None
            ),
            "fallback": row["fallback_num"] / n if n else None,
            "truncation": row["truncation_num"] / n if n else None,
            "mean_tokens": row["tokens_num"] / n if n else None,
        }
    )

aggregate = {
    "schema": suite_schema,
    "completed_datasets": [row["dataset"] for row in items],
    "datasets": items,
    "weighted_summary": weighted,
}
(root / "aggregate_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    f"# {suite_title}",
    "",
    f"- datasets: {suite_dataset_note}",
    "- methods: MLP-only h06 vs Attention-only h06",
    f"- completed_datasets: {len(items)}",
    "",
    "## Weighted Summary",
    "",
    "| method | correct / total | accuracy | MLP pruning | attention head pruning | fallback | truncation | mean tokens |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in weighted:
    mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
    lines.append(
        f"| `{row['method']}` | {int(row['correct'])} / {int(row['problems'])} | "
        f"{pct(row['accuracy'])} | {pct(row['mlp_pruning'])} | "
        f"{pct(row['attention_pruning'])} | {pct(row['fallback'])} | "
        f"{pct(row['truncation'])} | {mean_tokens} |"
    )
lines.extend(
    [
        "",
        "## By Dataset",
        "",
        "| dataset | method | correct / total | accuracy | protocol-valid acc | MLP pruning | attention head pruning | fallback | truncation | mean tokens | runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
)
for item in items:
    for row in item["rows"]:
        mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
        lines.append(
            f"| `{item['dataset']}` | `{row['method']}` | "
            f"{int(row['correct'])} / {int(row['problems'])} | {pct(row['accuracy'])} | "
            f"{pct(row['protocol_valid_accuracy'])} | {pct(row['mlp_pruning'])} | "
            f"{pct(row['attention_pruning'])} | {pct(row['fallback'])} | "
            f"{pct(row['truncation'])} | {mean_tokens} | `{row['runtime_backend'] or '-'}` |"
        )
(root / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

validate_same_rate_outputs() {
  "${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "aggregate_summary.json"
if not summary_path.exists():
    raise SystemExit(f"Missing aggregate summary: {summary_path}")
data = json.load(open(summary_path, "r", encoding="utf-8"))
datasets = {item["dataset"] for item in data.get("datasets", [])}
missing = {"gsm8k", "math500"} - datasets
if missing:
    raise SystemExit(f"Missing completed datasets: {sorted(missing)}")
for item in data.get("datasets", []):
    seen = {row["method"]: row for row in item.get("rows", [])}
    for method in ["mlp_only_stage_budget_h06", "attention_head_only_stage_budget_h06"]:
        if method not in seen:
            raise SystemExit(f"Missing method {method} for {item['dataset']}")
        runtime = seen[method].get("runtime_backend")
        if runtime == "dense_no_mask_v1" or not runtime:
            raise SystemExit(f"{item['dataset']} {method} used invalid runtime: {runtime}")
    attn = seen["attention_head_only_stage_budget_h06"].get("attention_pruning")
    if attn is None or float(attn) <= 0.0:
        raise SystemExit(f"{item['dataset']} attention pruning did not execute: {attn}")
    mlp = seen["mlp_only_stage_budget_h06"].get("mlp_pruning")
    if mlp is None or float(mlp) <= 0.0:
        raise SystemExit(f"{item['dataset']} MLP pruning did not execute: {mlp}")
print(f"Validation passed: {summary_path}")
PY
}

run_dataset() {
  local dataset="$1"
  local limit="$2"
  CONFIG="${CONFIG}" \
  RUN_ROOT="${RUN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  LOG_DIR="${LOG_ROOT}" \
  LOG_PREFIX="${LOG_PREFIX}" \
  STAGE_SEED="${SEED}" \
  STAGE_FINAL_SEEDS="${SEED}" \
  DATASETS_OVERRIDE="${dataset}" \
  STAGE_FINAL_EVAL_LIMIT="${limit}" \
  STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
  write_same_rate_aggregate
}

run_dataset gsm8k "${STAGE_ATTENTION_MLP_GSM8K_LIMIT:-520}"
run_dataset math500 "${STAGE_ATTENTION_MLP_MATH500_LIMIT:-520}"
validate_same_rate_outputs

echo "${SUITE_TITLE} complete: ${RUN_ROOT}/aggregate_summary.md"
