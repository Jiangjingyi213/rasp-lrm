# Stage-Budget + Output-Aware RASP v1 Configs

Generated configs for the v10 line.  This line intentionally avoids repeating the
old prior-name ablations and evaluates two mechanisms separately before combining
them:

- `budget_only_dev.yaml`: Phase A, stage budget controller with activation score.
- `output_aware_only_dev.yaml`: Phase B, output-aware score with fixed T30 budget.
- `combined_dev.yaml`: Phase C, global output-aware plus stage budget controller.
- `frozen_full.template.yaml`: frozen full-eval template after seed3/seed42 gates pass.
- `budget_v2_smoke.yaml`: Phase A2, two repaired budget grids on 128+128 smoke.
- `budget_v2_dev.yaml`: Phase A2, selected repaired budget on seed3 dev.
- `budget_v2_seed42.yaml`: Phase A2, frozen candidate confirmation on seed42 dev.
- `budget_v2_full.template.yaml`: Phase A2, frozen full-eval template.

All new results should live under `runs/10_stage_budget_output_aware_v1/`; logs
mirror to `logs/10_stage_budget_output_aware_v1/`.
