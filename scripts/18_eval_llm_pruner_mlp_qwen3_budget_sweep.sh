#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_llm_pruner_mlp_qwen3_budget_configs}"
GSM8K_LIMIT="${GSM8K_LIMIT:-1319}"
MATH500_LIMIT="${MATH500_LIMIT:-500}"
DATASETS="${DATASETS:-gsm8k math500}"
TAGS="${TAGS:-p05 p20 p40 p60}"
RUN_ROOT="${RUN_ROOT:-runs/llm_pruner_mlp_formal}"
CLEAN_RUN_DIR="${CLEAN_RUN_DIR:-1}"
LLM_PRUNER_IMPORTANCE="${LLM_PRUNER_IMPORTANCE:-l2}"
LLM_PRUNER_STRUCTURE="${LLM_PRUNER_STRUCTURE:-UL-UM}"
LLM_PRUNER_LAYERS="${LLM_PRUNER_LAYERS:-}"
LLM_PRUNER_PHYSICAL_PRUNING="${LLM_PRUNER_PHYSICAL_PRUNING:-true}"
ALLOW_UNSTABLE_LLM_PRUNER_MLP="${ALLOW_UNSTABLE_LLM_PRUNER_MLP:-0}"

if [[ "$ALLOW_UNSTABLE_LLM_PRUNER_MLP" != "1" ]]; then
  cat >&2 <<'EOF'
This LLM-Pruner-style Qwen3 MLP baseline is currently marked unstable.
Observed p05 runs collapsed into repeated-token generations, so it should not
be used as a formal baseline unless you are explicitly running diagnostics.

To run anyway, set:
  export ALLOW_UNSTABLE_LLM_PRUNER_MLP=1
EOF
  exit 2
fi

mkdir -p "$CONFIG_DIR"

"$PYTHON" - "$CONFIG_DIR" "$RUN_ROOT" "$GSM8K_LIMIT" "$MATH500_LIMIT" "$LLM_PRUNER_IMPORTANCE" "$LLM_PRUNER_STRUCTURE" "$LLM_PRUNER_LAYERS" "$LLM_PRUNER_PHYSICAL_PRUNING" <<'PY'
from pathlib import Path
import sys

config_dir = Path(sys.argv[1])
run_root = sys.argv[2].rstrip("/")
gsm8k_limit = int(sys.argv[3])
math500_limit = int(sys.argv[4])
importance = sys.argv[5]
structure = sys.argv[6]
layers = sys.argv[7].strip()
physical_pruning = sys.argv[8].lower() in {"1", "true", "yes", "y"}
config_dir.mkdir(parents=True, exist_ok=True)

base_model = """\
  name_or_path: Qwen/Qwen3-1.7B
  dtype: float32
  device_map: auto
  trust_remote_code: true
  attn_implementation: eager
"""

prompt = """\
prompt:
  use_chat_template: true
  enable_thinking: false
  system: You are a careful math reasoning assistant.
"""

budgets = {
    "p05": 0.05,
    "p20": 0.20,
    "p40": 0.40,
    "p60": 0.60,
}

def write_config(dataset: str, tag: str, ratio: float) -> None:
    importance_tag = importance.lower()
    structure_tag = structure.lower().replace("-", "")
    mode_tag = "physical" if physical_pruning else "mask"
    layer_tag = "all" if not layers else "layers" + layers.replace(",", "-").replace(" ", "")
    run_dir = f"{run_root}/eval_llm_pruner_mlp_{importance_tag}_{structure_tag}_{mode_tag}_{layer_tag}_{tag}_qwen3_{dataset}_budget"
    layers_yaml = ""
    if layers:
        parsed_layers = [int(x) for x in layers.split(",") if x.strip()]
        layers_yaml = "  llm_pruner_layers: [" + ", ".join(str(x) for x in parsed_layers) + "]\n"
    model = base_model + f"""\
  adapter: llm_pruner_mlp_qwen3
  llm_pruner_pruning_ratio: {ratio}
  llm_pruner_importance: {importance}
  llm_pruner_structure: {structure}
  llm_pruner_physical_pruning: {str(physical_pruning).lower()}
{layers_yaml}"""

    if dataset == "gsm8k":
        data = f"""\
data:
  dataset: gsm8k
  split: test
  limit: {gsm8k_limit}
"""
        max_new_tokens = 768
    elif dataset == "math500":
        data = f"""\
data:
  dataset: math500
  name_or_path: HuggingFaceH4/MATH-500
  split: test
  limit: {math500_limit}
"""
        max_new_tokens = 1024
    else:
        raise ValueError(dataset)

    text = f"""\
seed: 1

model:
{model}
{prompt}
{data}
generation:
  max_input_tokens: 2048
  max_new_tokens: {max_new_tokens}
  temperature: 0.0
  top_p: 1.0

paths:
  run_dir: {run_dir}
  trajectories: {run_dir}/01_trajectories.jsonl
  llm_pruner_mlp_summary: {run_dir}/00_llm_pruner_mlp_summary.json
"""
    (config_dir / f"{dataset}_{tag}.yaml").write_text(text, encoding="utf-8")

for dataset in ("gsm8k", "math500"):
    for tag, ratio in budgets.items():
        write_config(dataset, tag, ratio)
PY

for DATASET in $DATASETS; do
  for TAG in $TAGS; do
    CONFIG="$CONFIG_DIR/${DATASET}_${TAG}.yaml"
    RUN_DIR="$("$PYTHON" - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["paths"]["run_dir"])
PY
)"
    echo "==> $CONFIG"
    if [[ "$CLEAN_RUN_DIR" == "1" ]]; then
      rm -rf "$RUN_DIR"
    fi
    "$PYTHON" -m src.main_generate --config "$CONFIG"
  done
done

echo "LLM-Pruner-style MLP Qwen3 budget sweep complete."
