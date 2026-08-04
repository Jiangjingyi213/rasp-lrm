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
- `budget_v3_smoke.yaml`: Phase A3, actual-calibrated budget candidates on 128+128 smoke.
- `budget_v3_dev.yaml`: Phase A3, selected actual-calibrated budget on seed3 dev.
- `budget_v3_seed42.yaml`: Phase A3, frozen actual-calibrated confirmation on seed42 dev.
- `budget_v3_full.template.yaml`: Phase A3, frozen full-eval template.
- `budget_v4_smoke.yaml`: Phase A4, performance-guarded 32%+ candidates on 128+128 smoke.
- `budget_v4_dev.yaml`: Phase A4, selected 32%+ budget plus v3 under-pruned reference.
- `budget_v4_seed42.yaml`: Phase A4, frozen 32%+ confirmation on seed42 dev.
- `budget_v4_full.template.yaml`: Phase A4, frozen full-eval template.
- `budget_v41_smoke.yaml`: Phase A4.1, stage-floor lift candidates after v4 undershot 32%.
- `budget_v41_dev.yaml`: Phase A4.1, selected stage-lift candidate plus v4 mild reference.
- `budget_v41_seed42.yaml`: Phase A4.1, frozen stage-lift confirmation on seed42 dev.
- `budget_v41_full.template.yaml`: Phase A4.1, frozen full-eval template.

All new results should live under `runs/10_stage_budget_output_aware_v1/`; logs
mirror to `logs/10_stage_budget_output_aware_v1/`.
