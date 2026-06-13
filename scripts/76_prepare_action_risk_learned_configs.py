#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = os.environ.get(
    "ACTION_RISK_LEARNED_ROOT",
    "runs/07_stage_aware/08_action_risk_learned_single_window_pilot",
)
CONFIG_DIR = ROOT / "configs/generated_action_risk_learned_pilot"
CHECKPOINT = f"{RUN_ROOT}/checkpoints/action_risk_single_window.joblib"
CALIBRATION = Path(RUN_ROOT) / "checkpoints/calibration_summary.json"
OPERATING_POINTS = ("conservative", "balanced", "aggressive")
RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def base_config(dataset: str, limit: int) -> dict:
    data = {"dataset": dataset, "split": "test", "limit": limit}
    if dataset == "math500":
        data["name_or_path"] = "HuggingFaceH4/MATH-500"
    return {
        "seed": 1,
        "model": {
            "name_or_path": "Qwen/Qwen3-1.7B",
            "dtype": "float32",
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        },
        "prompt": {
            "use_chat_template": True,
            "enable_thinking": False,
            "system": "You are a careful math reasoning assistant.",
        },
        "data": data,
        "generation": {
            "max_input_tokens": 2048,
            "max_new_tokens": 768,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def main() -> None:
    gpu_count = int(os.environ.get("ACTION_RISK_LEARNED_GPU_COUNT", "8"))
    limit = int(os.environ.get("ACTION_RISK_LEARNED_LIMIT", "100"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    queue = 0
    for dataset in ("gsm8k", "math500"):
        jobs = [("dense", None, None)]
        for point in OPERATING_POINTS:
            jobs.append((f"context_only_{point}", "context_only", point))
            if calibration["operating_points"][point].get("hidden_veto_eligible", False):
                jobs.append((f"hidden_veto_{point}", "hidden_veto", point))
        for tag, variant, point in jobs:
            cfg = base_config(dataset, limit)
            runtime = {
                "backend": "logical_mask_v0",
                "controller": "fixed" if variant is None else "action_risk_single_window",
                "fixed_ratio": 0.0,
                "ratios": RATIOS,
                "window_tokens": 16,
            }
            if variant is not None:
                runtime.update(
                    {
                        "policy_checkpoint": CHECKPOINT,
                        "policy_variant": variant,
                        "operating_point": point,
                        "eligible_boundaries": [32, 96, 160],
                        "max_action_windows": 1,
                    }
                )
            cfg["runtime_rasp"] = runtime
            run_dir = f"{RUN_ROOT}/online/{dataset}/{tag}"
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "runtime_summary": f"{run_dir}/00_runtime_summary.json",
            }
            path = CONFIG_DIR / f"{dataset}_{tag}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            manifest.append(
                {
                    "gpu_queue": queue,
                    "dataset": dataset,
                    "tag": tag,
                    "policy_variant": variant,
                    "operating_point": point,
                    "config": str(path.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            queue = (queue + 1) % gpu_count
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index in range(gpu_count):
        configs = [row["config"] for row in manifest if row["gpu_queue"] == index]
        (CONFIG_DIR / f"gpu{index}.list").write_text("\n".join(configs) + "\n", encoding="utf-8")
    print(f"generated {len(manifest)} learned Action-Risk jobs across {gpu_count} queues")


if __name__ == "__main__":
    main()
