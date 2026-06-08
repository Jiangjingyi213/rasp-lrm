#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/generated_rasp_phase_b_aligned_bank"
RUN_ROOT = "runs/rasp_phase_b_aligned_bank"
SOURCES = (
    ("gsm8k_train", "configs/exp_rasp_phase_b_aligned_bank_gsm8k_train.yaml"),
    ("math_train", "configs/exp_rasp_phase_b_aligned_bank_math_train.yaml"),
)


def main() -> None:
    total = int(os.environ.get("RASP_PHASE_B_LIMIT_PER_SOURCE", "500"))
    shard_size = int(os.environ.get("RASP_PHASE_B_SHARD_SIZE", "25"))
    gpu_count = int(os.environ.get("RASP_PHASE_B_GPU_COUNT", "8"))
    max_boundaries = int(os.environ.get("RASP_PHASE_B_MAX_BOUNDARIES_PER_EXAMPLE", "12"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    gpu = 0
    for source, template in SOURCES:
        with (ROOT / template).open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        for shard, offset in enumerate(range(0, total, shard_size)):
            cfg = copy.deepcopy(base)
            cfg["data"]["limit"] = min(shard_size, total - offset)
            cfg["data"]["offset"] = offset
            if max_boundaries > 0:
                cfg["aligned_window_bank"]["max_boundaries_per_example"] = max_boundaries
            else:
                cfg["aligned_window_bank"].pop("max_boundaries_per_example", None)
            run_dir = f"{RUN_ROOT}/{source}_s{shard:02d}"
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "counterfactuals": f"{run_dir}/03_aligned_window_counterfactuals.jsonl",
                "probe_dataset": f"{run_dir}/05_probe_dataset.jsonl",
                "probe_hidden_states": f"{run_dir}/05_probe_hidden_states.pt",
                "aligned_window_bank_summary": f"{run_dir}/06_aligned_window_bank_summary.json",
                "aligned_window_bank_validation": f"{run_dir}/07_aligned_window_bank_validation.json",
            }
            path = CONFIG_DIR / f"{source}_s{shard:02d}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            manifest.append({"gpu_queue": gpu, "config": str(path.relative_to(ROOT)), "run_dir": run_dir})
            gpu = (gpu + 1) % gpu_count
    with (CONFIG_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for index in range(gpu_count):
        with (CONFIG_DIR / f"gpu{index}.list").open("w", encoding="utf-8") as handle:
            for row in manifest:
                if row["gpu_queue"] == index:
                    handle.write(row["config"] + "\n")
    print(f"generated {len(manifest)} Phase B shards across {gpu_count} GPUs")


if __name__ == "__main__":
    main()
