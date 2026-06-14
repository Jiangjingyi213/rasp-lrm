#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = os.environ.get(
    "FULL_TRAJECTORY_ROOT",
    "runs/07_stage_aware/10_full_trajectory_multi_window",
)
FIXED_ROOT = Path(WORKFLOW_ROOT) / "03_fixed_multi_window_dev"
RUN_ROOT = Path(WORKFLOW_ROOT) / "04_on_policy_smoke"
CONFIG_DIR = ROOT / "configs/generated_on_policy_smoke"


def main() -> None:
    gpu_count = int(os.environ.get("ON_POLICY_GPU_COUNT", "2"))
    if gpu_count < 1:
        raise ValueError("ON_POLICY_GPU_COUNT must be positive")
    selection = json.loads(
        (FIXED_ROOT / "selected_behavior_policy.json").read_text(encoding="utf-8")
    )
    tag = selection.get("selected_behavior_policy")
    if not tag or not selection.get("allowed_for_on_policy_smoke"):
        raise SystemExit("No behavior policy passed the multi-window feasibility gate")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for queue, dataset in enumerate(("gsm8k", "math_train")):
        source_config = ROOT / "configs/generated_fixed_multi_window_dev" / f"{dataset}_{tag}.yaml"
        cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        cfg["on_policy_bank"] = {
            "behavior_policy_tag": tag,
            "candidate_ratios": [0.0, 0.10, 0.20, 0.30],
            "window_tokens": 16,
            "cooldown_tokens": 16,
            "decision_start": 32,
            "decision_stride": 32,
            "max_problems": 4,
            "max_boundaries_per_problem": 2,
            "require_dense_and_behavior_correct": True,
            "replay_entropy_tolerance": 1e-5,
            "replay_confidence_tolerance": 1e-5,
            "replay_hidden_cosine_tolerance": 1e-5,
            "replay_logits_tolerance": 1e-4,
        }
        cfg["stage_sensitivity"] = {
            "checkpoint": "runs/07_stage_aware/03_s1_three_stage_probe/seed_1/hidden_pca_nonlinear/stage_probe.pt",
            "reasoning_threshold": 0.8069186806678772,
            "confidence_threshold": 0.8069186806678772,
            "recent_tokens": 128,
            "require_causal_features": True,
        }
        run_dir = RUN_ROOT / dataset
        cfg["paths"] = {
            "run_dir": str(run_dir),
            "dense_trajectories": str(FIXED_ROOT / "online" / dataset / "dense" / "01_trajectories.jsonl"),
            "behavior_trajectories": str(FIXED_ROOT / "online" / dataset / tag / "01_trajectories.jsonl"),
            "on_policy_dataset": str(run_dir / "01_on_policy_dataset.jsonl"),
            "on_policy_hidden_states": str(run_dir / "01_on_policy_hidden_states.pt"),
            "on_policy_summary": str(run_dir / "02_on_policy_summary.json"),
            "on_policy_validation": str(run_dir / "03_on_policy_validation.json"),
        }
        path = CONFIG_DIR / f"{dataset}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        manifest.append(
            {
                "gpu_queue": queue % gpu_count,
                "dataset": dataset,
                "behavior_policy_tag": tag,
                "config": str(path.relative_to(ROOT)),
                "run_dir": str(run_dir),
            }
        )
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"generated on-policy smoke configs for behavior policy {tag}")


if __name__ == "__main__":
    main()
