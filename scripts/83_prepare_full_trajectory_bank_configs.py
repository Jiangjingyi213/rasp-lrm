#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = os.environ.get(
    "FULL_TRAJECTORY_ROOT",
    "runs/07_stage_aware/10_full_trajectory_multi_window",
)
PROFILE = os.environ.get("PROFILE", "dense_smoke")
if PROFILE not in {"dense_smoke", "dense_pilot"}:
    raise SystemExit(f"Unsupported full-trajectory bank profile: {PROFILE}")

CONFIG_DIR = ROOT / "configs" / f"generated_full_trajectory_{PROFILE}"
PROFILE_DEFAULTS = {
    "dense_smoke": {
        "ratios": [0.0, 0.20, 0.50],
        "inputs": {"gsm8k_train": 8, "math_train": 12},
        "offset": 0,
        "shard_size": 2,
        "max_dense_replay_flip_rate": 0.0,
    },
    "dense_pilot": {
        "ratios": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
        "inputs": {"gsm8k_train": 32, "math_train": 48},
        "offset": 100,
        "shard_size": 5,
        "max_dense_replay_flip_rate": 0.01,
    },
}
PROFILE_DIRS = {
    "dense_smoke": "01_dense_bank_smoke",
    "dense_pilot": "02_dense_bank_pilot",
}
SOURCES = (
    ("gsm8k_train", "configs/exp_rasp_phase_b_aligned_bank_gsm8k_train.yaml"),
    ("math_train", "configs/exp_rasp_phase_b_aligned_bank_math_train.yaml"),
)
STAGE_CHECKPOINT = os.environ.get(
    "STAGE_PROBE_CHECKPOINT",
    "runs/07_stage_aware/03_s1_three_stage_probe/seed_1/hidden_pca_nonlinear/stage_probe.pt",
)


def main() -> None:
    defaults = PROFILE_DEFAULTS[PROFILE]
    gpu_count = int(os.environ.get("FULL_TRAJECTORY_GPU_COUNT", "8"))
    shard_size = int(os.environ.get("FULL_TRAJECTORY_SHARD_SIZE", defaults["shard_size"]))
    if not (ROOT / STAGE_CHECKPOINT).exists():
        raise SystemExit(f"Missing causal hidden stage checkpoint: {STAGE_CHECKPOINT}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    queue = 0
    for source, template in SOURCES:
        total = int(
            os.environ.get(
                f"FULL_TRAJECTORY_{PROFILE.upper()}_{source.upper()}_INPUTS",
                defaults["inputs"][source],
            )
        )
        base_offset = int(os.environ.get(f"FULL_TRAJECTORY_{PROFILE.upper()}_OFFSET", defaults["offset"]))
        with (ROOT / template).open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        for shard, local_offset in enumerate(range(0, total, shard_size)):
            cfg = copy.deepcopy(base)
            cfg["data"]["limit"] = min(shard_size, total - local_offset)
            cfg["data"]["offset"] = base_offset + local_offset
            cfg["generation"]["overwrite_output"] = True
            cfg["generation"]["store_generated_token_ids"] = True
            cfg["aligned_window_bank"] = {
                "ratios": defaults["ratios"],
                "window_tokens": 16,
                "boundary_sampling": "causal_grid",
                "decision_start": 32,
                "decision_stride": 32,
                "include_tail_anchor": True,
                "max_dense_replay_flip_rate": defaults["max_dense_replay_flip_rate"],
            }
            cfg["stage_sensitivity"] = {
                "checkpoint": STAGE_CHECKPOINT,
                "reasoning_threshold": 0.8069186806678772,
                "confidence_threshold": 0.8069186806678772,
                "recent_tokens": 128,
                "measure_all_operational_stages": True,
                "diagnostic_only": True,
                "require_causal_features": True,
            }
            run_dir = f"{WORKFLOW_ROOT}/{PROFILE_DIRS[PROFILE]}/bank/{source}_s{shard:02d}"
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "counterfactuals": f"{run_dir}/03_full_trajectory_counterfactuals.jsonl",
                "probe_dataset": f"{run_dir}/05_full_trajectory_probe_dataset.jsonl",
                "probe_hidden_states": f"{run_dir}/05_full_trajectory_hidden_states.pt",
                "aligned_window_bank_summary": f"{run_dir}/06_full_trajectory_bank_summary.json",
                "aligned_window_bank_validation": f"{run_dir}/07_full_trajectory_bank_validation.json",
            }
            path = CONFIG_DIR / f"{source}_s{shard:02d}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            manifest.append(
                {
                    "gpu_queue": queue,
                    "profile": PROFILE,
                    "source": source,
                    "offset": cfg["data"]["offset"],
                    "input_tasks": cfg["data"]["limit"],
                    "config": str(path.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            queue = (queue + 1) % gpu_count
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index in range(gpu_count):
        configs = [item["config"] for item in manifest if item["gpu_queue"] == index]
        (CONFIG_DIR / f"gpu{index}.list").write_text("\n".join(configs) + "\n", encoding="utf-8")
    print(f"generated {len(manifest)} {PROFILE} shards across {gpu_count} queues")


if __name__ == "__main__":
    main()
