# Reasoning-aware Dynamic Structured Pruning

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

The default config is `configs/exp_minimal_gsm8k.yaml` and writes all artifacts under `runs/minimal_gsm8k_qwen1_5b/`.
The Qwen3 motivation config is `configs/exp_motivation_qwen3_gsm8k.yaml` and writes under `runs/motivation_qwen3_gsm8k_l8_no_l0/`.
The larger Qwen3 overnight config is `configs/exp_motivation_qwen3_gsm8k_l32.yaml`.

## Key Outputs

- `01_trajectories.jsonl`: dense generations with extracted answers.
- `02_segments.jsonl`: segmented reasoning traces.
- `03_counterfactuals.jsonl`: `(example, segment, layer)` answer-flip heatmap rows.
- `03_counterfactuals.oracles.json`: static/prompt/step oracle summary.
- `04_entropy_auc.json`: entropy as a pruning-risk predictor.
- `05_probe_metrics.json`: hidden-state probe validation metrics.
- `06_heatmap_summary.json`: layer/segment flip-rate summaries for motivation plots.
