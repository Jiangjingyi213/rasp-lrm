# LLM-Pruner-Style Qwen3 MLP Baseline

This baseline is a Qwen3 MLP-only static width-pruning port inspired by
`horseee/LLM-Pruner`. It is intended as a conservative static structured
pruning baseline for GSM8K/MATH500 evaluation.

## Scope

- Target model: `Qwen/Qwen3-1.7B`
- Target module: MLP intermediate channels
- Pruned group: one matched intermediate channel across `gate_proj`,
  `up_proj`, and `down_proj`
- Physical pruning: configurable
- Recovery/LoRA: no
- Attention pruning: no

This is not a full official LLM-Pruner port. It does not implement the original
dependency graph, Taylor importance, or post-pruning LoRA recovery. Reports
should refer to it as `LLM-Pruner-style MLP static width pruning` unless those
extra components are added later.

## Importance

The current implementation supports:

- `l1`: group L1 norm over `gate_proj`, `up_proj`, and `down_proj`
- `l2`: group L2 norm over `gate_proj`, `up_proj`, and `down_proj`
- `random`: random group ranking for sanity checks

Default: `l2`.

## Structure

The current implementation supports:

- `UL-UM`: uniform layer, uniform module; each selected layer prunes the same
  ratio of MLP channels
- `AL-AM`: adaptive layer/module allocation using standardized global scores

Default: `UL-UM`.

## Diagnostic Options

Because Qwen3 can be fragile under all-layer physical MLP width pruning, the
sweep script exposes two diagnostics:

- `LLM_PRUNER_PHYSICAL_PRUNING=false`: zero the selected MLP channels while
  keeping tensor shapes unchanged. This distinguishes pruning-policy failure
  from physical-shape/export issues.
- `LLM_PRUNER_LAYERS=4,5,...`: prune only selected layers. This mirrors the
  common LLM-Pruner practice of avoiding the earliest layers during block-wise
  pruning.

## Formal Sweep

Use:

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export LLM_PRUNER_PHYSICAL_PRUNING=false
export LLM_PRUNER_LAYERS=4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27

nohup bash scripts/18_eval_llm_pruner_mlp_qwen3_budget_sweep.sh \
  > logs/llm_pruner_mlp_qwen3_gpu0.log 2>&1 &
```

Default budgets:

```text
p05, p20, p40, p60
```

Default outputs:

```text
runs/llm_pruner_mlp_formal/eval_llm_pruner_mlp_l2_ulum_p05_qwen3_gsm8k_budget/
runs/llm_pruner_mlp_formal/eval_llm_pruner_mlp_l2_ulum_p20_qwen3_gsm8k_budget/
...
```

Each run contains:

```text
00_llm_pruner_mlp_summary.json
01_trajectories.jsonl
```
