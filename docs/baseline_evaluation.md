# Baseline Evaluation Plan

This document separates two baseline families for the RASP motivation and method experiments.

## Offline Policy Baselines

Offline policy baselines reuse the existing `03_counterfactuals.jsonl` tables. They do not create a new pruned model. Each policy selects one pruning action per `(dataset, problem, segment)` from the already evaluated action space and reports the realized answer-flip risk.

Supported by `src.main_offline_baselines`:

- `dense_lrm`: no pruning; flip risk is zero and pruning strength is zero.
- `static_<module>_r<ratio>`: one fixed action for every problem-step.
- `static_best_safe_oracle`: one globally safest action chosen from the counterfactual table.
- `entropy_quantile_policy`: entropy quantiles route high-risk steps to no/low pruning and low-risk steps to stronger MLP pruning.
- `confidence_quantile_policy`: uses `1 - confidence` with the same routing rule.
- `prompt_router_safe_oracle`: one safest action per problem; this is an upper bound for prompt-level routing.
- `step_safe_oracle`: one safest action per problem-step; this is the offline upper bound.

Primary metrics:

- `selected_action_flip_rate`: lower is safer.
- `average_pruning_ratio`: average selected pruning ratio; dense uses `0`.
- `average_pruning_strength_proxy`: module-weighted pruning proxy, not a measured speedup.
- `gap_to_step_oracle`: selected flip rate minus step oracle flip rate.
- module, choice, stage, and dataset breakdowns.

Example:

```bash
python3 -m src.main_offline_baselines \
  --configs /tmp/formal_math500_full_s0.yaml /tmp/formal_math500_full_s1.yaml \
  --output runs/formal_qwen3_math500_offline_baselines.json \
  --summary-csv runs/formal_qwen3_math500_offline_baselines.csv \
  --selected-output runs/formal_qwen3_math500_offline_selected_actions.jsonl
```

When temporary configs are unavailable, pass counterfactual JSONL files directly:

```bash
python3 -m src.main_offline_baselines \
  --inputs runs/formal_qwen3_math500_full_s0/03_counterfactuals.jsonl \
           runs/formal_qwen3_math500_full_s1/03_counterfactuals.jsonl \
  --output runs/formal_qwen3_math500_offline_baselines.json \
  --summary-csv runs/formal_qwen3_math500_offline_baselines.csv
```

## External Pruned-Model Baselines

External pruning baselines, such as LLM-Pruner and FLAP, should be reproduced outside this repository first. They are not equivalent to the offline policy baselines because they produce an actual pruned model.

Recommended protocol:

1. Clone each external repository outside this project, for example under `/home/cike/jjy/external/`.
2. Use a separate conda environment so `rasp_qwen3` remains stable.
3. Confirm whether the method supports `Qwen/Qwen3-1.7B`; if not, avoid large compatibility rewrites until the RASP-Zero direction is validated.
4. Export a HuggingFace-compatible pruned model directory.
5. Evaluate the pruned model with this repository's `main_generate` and the same prompts, decoding config, answer parser, and datasets.

Report:

- dense accuracy after pruning;
- accuracy drop relative to dense Qwen3;
- pruning ratio or remaining parameter ratio;
- peak memory and wall-clock/tokens-per-second if measured;
- whether calibration data, recovery training, or post-pruning tuning was used.

Current caution:

- LLM-Pruner and FLAP are mainly documented for LLaMA-family/Vicuna-style models. Qwen3 support should be treated as unknown until verified.
- GRIFFIN-LRM needs an exact paper/repository reference before it can be placed into the implementation plan.
