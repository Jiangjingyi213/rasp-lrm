#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = os.environ.get(
    "STAGE_ACTION_RISK_V2_ROOT",
    "runs/07_stage_aware/09_stage_action_risk_v2",
)
CONFIG_DIR = ROOT / "configs/generated_stage_action_risk_v2"
RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BOUNDARIES = [32, 96, 160]
STAGE_CHECKPOINT = os.environ.get(
    "STAGE_PROBE_CHECKPOINT",
    "runs/07_stage_aware/03_s1_three_stage_probe/seed_1/hidden_pca_nonlinear/stage_probe.pt",
)
STAGE_THRESHOLD = float(os.environ.get("STAGE_REASONING_THRESHOLD", "0.8069186806678772"))
SOURCES = (
    ("gsm8k_train", "configs/exp_rasp_phase_b_aligned_bank_gsm8k_train.yaml", 220),
    ("math_train", "configs/exp_rasp_phase_b_aligned_bank_math_train.yaml", 320),
)


def main() -> None:
    gpu_count = int(os.environ.get("STAGE_ACTION_RISK_V2_GPU_COUNT", "8"))
    shard_size = int(os.environ.get("STAGE_ACTION_RISK_V2_SHARD_SIZE", "10"))
    if not (ROOT / STAGE_CHECKPOINT).exists():
        raise SystemExit(f"Missing stage checkpoint: {STAGE_CHECKPOINT}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    queue = 0
    for source, template, default_total in SOURCES:
        total = int(os.environ.get(f"STAGE_ACTION_RISK_V2_{source.upper()}_INPUTS", default_total))
        with (ROOT / template).open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        for shard, offset in enumerate(range(0, total, shard_size)):
            cfg = copy.deepcopy(base)
            cfg["data"]["limit"] = min(shard_size, total - offset)
            cfg["data"]["offset"] = offset
            cfg["generation"]["overwrite_output"] = True
            cfg["aligned_window_bank"].update(
                {
                    "ratios": RATIOS,
                    "window_tokens": 16,
                    "boundary_positions": BOUNDARIES,
                    "boundary_sampling": "explicit_positions",
                }
            )
            cfg["aligned_window_bank"].pop("max_boundaries_per_example", None)
            cfg["stage_sensitivity"] = {
                "checkpoint": STAGE_CHECKPOINT,
                "reasoning_threshold": STAGE_THRESHOLD,
                "recent_tokens": 128,
                "measure_all_operational_stages": True,
                "diagnostic_only": True,
                "s1_5_controller_gate_passed": False,
            }
            run_dir = f"{RUN_ROOT}/bank/{source}_s{shard:02d}"
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "counterfactuals": f"{run_dir}/03_stage_action_counterfactuals.jsonl",
                "probe_dataset": f"{run_dir}/05_stage_action_probe_dataset.jsonl",
                "probe_hidden_states": f"{run_dir}/05_stage_action_probe_hidden_states.pt",
                "aligned_window_bank_summary": f"{run_dir}/06_stage_action_bank_summary.json",
                "aligned_window_bank_validation": f"{run_dir}/07_stage_action_bank_validation.json",
            }
            path = CONFIG_DIR / f"{source}_s{shard:02d}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            manifest.append(
                {
                    "gpu_queue": queue,
                    "source": source,
                    "offset": offset,
                    "config": str(path.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            queue = (queue + 1) % gpu_count
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index in range(gpu_count):
        configs = [row["config"] for row in manifest if row["gpu_queue"] == index]
        (CONFIG_DIR / f"gpu{index}.list").write_text("\n".join(configs) + "\n", encoding="utf-8")
    print(f"generated {len(manifest)} Stage-Action-Risk v2 shards across {gpu_count} queues")


if __name__ == "__main__":
    main()
