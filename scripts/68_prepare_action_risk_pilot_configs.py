#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = os.environ.get("ACTION_RISK_RUN_ROOT", "runs/07_stage_aware/06_action_risk_pilot")
CONFIG_DIR = ROOT / "configs/generated_action_risk_pilot"
RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
SOURCES = (
    ("gsm8k_train", "configs/exp_rasp_phase_b_aligned_bank_gsm8k_train.yaml", 140),
    ("math_train", "configs/exp_rasp_phase_b_aligned_bank_math_train.yaml", 200),
)


def main() -> None:
    gpu_count = int(os.environ.get("ACTION_RISK_GPU_COUNT", "4"))
    shard_size = int(os.environ.get("ACTION_RISK_SHARD_SIZE", "10"))
    max_boundaries = int(os.environ.get("ACTION_RISK_MAX_BOUNDARIES_PER_EXAMPLE", "12"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    queue = 0
    for source, template, default_total in SOURCES:
        total = int(os.environ.get(f"ACTION_RISK_{source.upper()}_INPUTS", str(default_total)))
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
                    "max_boundaries_per_example": max_boundaries,
                    "boundary_sampling": "uniform_full_trajectory",
                }
            )
            run_dir = f"{RUN_ROOT}/bank/{source}_s{shard:02d}"
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "counterfactuals": f"{run_dir}/03_action_window_counterfactuals.jsonl",
                "probe_dataset": f"{run_dir}/05_action_probe_dataset.jsonl",
                "probe_hidden_states": f"{run_dir}/05_action_probe_hidden_states.pt",
                "aligned_window_bank_summary": f"{run_dir}/06_action_window_bank_summary.json",
                "aligned_window_bank_validation": f"{run_dir}/07_action_window_bank_validation.json",
            }
            path = CONFIG_DIR / f"{source}_s{shard:02d}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            manifest.append(
                {
                    "gpu_queue": queue,
                    "source": source,
                    "offset": offset,
                    "input_tasks": cfg["data"]["limit"],
                    "config": str(path.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            queue = (queue + 1) % gpu_count
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index in range(gpu_count):
        rows = [row["config"] for row in manifest if row["gpu_queue"] == index]
        (CONFIG_DIR / f"gpu{index}.list").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"generated {len(manifest)} action-risk shards across {gpu_count} queues")


if __name__ == "__main__":
    main()
