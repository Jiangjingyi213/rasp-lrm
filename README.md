# Reasoning-aware Dynamic Structured Pruning

## Current Mainline

The active research workflow is explicit reasoning-stage calibrated structured
pruning:

```text
decontaminated Big-Math problems
-> Qwen3 self-generated explicit-stage trajectories
-> stage-conditioned WIFV calibration
-> frozen structured MLP masks
-> explicit stage-triggered runtime mask switching
-> held-out GSM8K / MATH-500 evaluation
```

Run the centralized smoke workflow on the remote server:

```bash
PYTHON=/home/cike/jjy/envs/rasp_qwen3_eval/bin/python \
bash scripts/preflight_stage_calibrated_pruning.sh

PROFILE=smoke PYTHON=/home/cike/jjy/envs/rasp_qwen3_eval/bin/python \
bash scripts/run_stage_calibrated_pruning.sh
```

See `docs/CURRENT_WORKFLOW_ZH.md`. The older action-risk and multi-window
workflows are frozen legacy experiments and are no longer the default route.

This repository contains a minimal closed-loop experiment for motivating reasoning-aware structured pruning in LRMs.

## Experiment Chain

1. Generate dense GSM8K/MATH500 reasoning trajectories.
2. Split each trajectory into rule-based reasoning segments.
3. Run segment-level counterfactual structured pruning and produce an answer-flip heatmap.
4. Compare static, prompt, and step oracle flip rates.
5. Measure entropy vs pruning risk ROC/PR AUC.
6. Train a linear hidden-state pruning-risk probe.
7. Summarize layer/segment heatmaps for motivation plots.

The first implementation uses layer skip plus optional attention/MLP block ablations as structured pruning interventions. This is intentionally coarse so the first pass is easy to validate; head/neuron/group pruning can be added behind the same output schema.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/01_generate_trajectories.sh
bash scripts/02_segment_trajectories.sh
bash scripts/03_counterfactual_prune.sh
bash scripts/04_entropy_auc.sh
bash scripts/05_train_probe.sh
bash scripts/06_heatmap_summary.sh
```

Offline baseline policies can be evaluated from finished counterfactual tables:

```bash
python3 -m src.main_offline_baselines \
  --inputs runs/01_motivation/formal_qwen3_math500_full_s0/03_counterfactuals.jsonl \
           runs/01_motivation/formal_qwen3_math500_full_s1/03_counterfactuals.jsonl \
  --output runs/01_motivation/formal_qwen3_math500_offline_baselines.json \
  --summary-csv runs/01_motivation/formal_qwen3_math500_offline_baselines.csv
```

For the first reasoning-aware policy simulation over the completed GSM8K and
MATH500 counterfactual tables, run:

```bash
bash scripts/19_rasp_zero_offline.sh
```

This generates problem-level out-of-fold hidden-state risk scores and evaluates
budget-matched static, entropy-only, confidence-only, probe-only, and
stage-aware RASP-Zero policies. See `docs/rasp_zero_offline.md`.

The second offline stage adds action-conditioned risk prediction and a
multi-module router:

```bash
bash scripts/21_rasp_zero_offline_v2.sh
```

See `docs/rasp_zero_offline_v2_zh.md` for the detailed Chinese experiment
record and interpretation guide.

The current trainable runtime-ratio prototype is RASP-Train v2.1 shared action-risk learning:

```bash
bash scripts/35_prepare_rasp_train_v1_data.sh
bash scripts/36_train_rasp_train_v1.sh
bash scripts/37_eval_rasp_train_v1_offline.sh
```

It trains one budget-independent risk model, calibrates separate B15/B20
thresholds with problem-level fold-stability checks, and reports offline
results on separate test problems. It must pass offline validation before
`scripts/38_eval_rasp_train_v1_online_smoke.sh` is run. The
online smoke includes a paired ratio-zero control and writes
`14_paired_dense_comparison.json`. v2.1 outputs are written under
`runs/04_rasp_train/01_legacy/rasp_train_v2_1/`; prior v1/v2 results remain unchanged. See
`docs/rasp_train_v1_zh.md`.

See `docs/baseline_evaluation.md` for the distinction between offline policy baselines and external pruned-model baselines such as LLM-Pruner/FLAP.

External pruning repositories are cloned on the remote server, not committed here:

```bash
bash baselines/scripts/clone_external_baselines.sh
bash baselines/scripts/run_flap_smoke.sh
bash baselines/scripts/run_llm_pruner_smoke.sh
bash baselines/scripts/run_griffin_smoke.sh
```

Smoke reports are written under `runs/02_baselines/external_baselines/`.

For the Qwen3 motivation pipeline:

```bash
CONFIG=configs/exp_motivation_qwen3_gsm8k.yaml
bash scripts/01_generate_trajectories.sh "$CONFIG"
bash scripts/02_segment_trajectories.sh "$CONFIG"
bash scripts/03_counterfactual_prune.sh "$CONFIG"
bash scripts/04_entropy_auc.sh "$CONFIG"
bash scripts/05_train_probe.sh "$CONFIG"
bash scripts/06_heatmap_summary.sh "$CONFIG"
```

The default config is `configs/exp_minimal_gsm8k.yaml` and writes all artifacts under `runs/01_motivation/minimal_gsm8k_qwen1_5b/`.
The Qwen3 motivation config is `configs/exp_motivation_qwen3_gsm8k.yaml` and writes under `runs/01_motivation/motivation_qwen3_gsm8k_l8_no_l0/`.
The larger Qwen3 overnight config is `configs/exp_motivation_qwen3_gsm8k_l32.yaml`.
No-layer-20 configs are available for `gsm8k`, `math500`, and `aime2024`; run them separately, then combine summaries with `scripts/07_collect_results.sh`.

## Key Outputs

- `01_trajectories.jsonl`: dense generations with extracted answers.
- `02_segments.jsonl`: segmented reasoning traces.
- `03_counterfactuals.jsonl`: `(example, segment, layer)` answer-flip heatmap rows.
- `03_counterfactuals.oracles.json`: static/prompt/step oracle summary.
- `04_entropy_auc.json`: entropy as a pruning-risk predictor.
- `05_probe_metrics.json`: hidden-state probe validation metrics.
- `06_heatmap_summary.json`: layer/segment flip-rate summaries for motivation plots.
- `07_offline_baselines.json`: safety-oriented offline policy baseline summary from existing counterfactual rows.
