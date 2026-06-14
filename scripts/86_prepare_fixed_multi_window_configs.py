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
RUN_ROOT = f"{WORKFLOW_ROOT}/03_fixed_multi_window_dev"
CONFIG_DIR = ROOT / "configs/generated_fixed_multi_window_dev"
LIMIT = int(os.environ.get("MULTI_WINDOW_DEV_LIMIT", "20"))
OFFSET = int(os.environ.get("MULTI_WINDOW_DEV_OFFSET", "2000"))
POLICIES = (
    ("r010_c64_m2", 0.10, 64, 2),
    ("r010_c32_m4", 0.10, 32, 4),
    ("r020_c64_m2", 0.20, 64, 2),
    ("r020_c32_m4", 0.20, 32, 4),
    ("r030_c64_m2", 0.30, 64, 2),
)


def base_config(source: str) -> dict:
    if source == "gsm8k":
        data = {"dataset": "gsm8k", "split": "train", "offset": OFFSET, "limit": LIMIT}
    else:
        data = {
            "dataset": "math500",
            "dataset_label": "math_train",
            "name_or_path": "rasbt/math_full_minus_math500",
            "split": "train",
            "offset": OFFSET,
            "limit": LIMIT,
        }
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
    gpu_count = int(os.environ.get("MULTI_WINDOW_GPU_COUNT", "8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    queue = 0
    for dataset in ("gsm8k", "math_train"):
        jobs = [("dense", None, None, None), *POLICIES]
        for tag, ratio, cadence, max_windows in jobs:
            cfg = base_config(dataset)
            runtime = {
                "backend": "logical_mask_v0",
                "controller": "fixed" if tag == "dense" else "fixed_multi_window",
                "fixed_ratio": 0.0 if ratio is None else ratio,
                "ratios": [0.0, 0.10, 0.20, 0.30],
                "window_tokens": 16,
                "store_router_hidden_states": True,
            }
            if tag != "dense":
                runtime.update(
                    {
                        "decision_start": 32,
                        "cadence_tokens": cadence,
                        "max_windows": max_windows,
                        "cooldown_tokens": 16,
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
                    "ratio": ratio,
                    "cadence_tokens": cadence,
                    "max_windows": max_windows,
                    "config": str(path.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            queue = (queue + 1) % gpu_count
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index in range(gpu_count):
        configs = [item["config"] for item in manifest if item["gpu_queue"] == index]
        (CONFIG_DIR / f"gpu{index}.list").write_text("\n".join(configs) + "\n", encoding="utf-8")
    Path(RUN_ROOT).mkdir(parents=True, exist_ok=True)
    (Path(RUN_ROOT) / "dev_split_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rasp_multi_window_dev_split_v1",
                "sources": ["gsm8k_train", "math_full_minus_math500_train"],
                "offset": OFFSET,
                "limit": LIMIT,
                "final_test_sources_used": False,
                "disjoint_from_dense_bank_offsets": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"generated {len(manifest)} fixed multi-window jobs across {gpu_count} queues")


if __name__ == "__main__":
    main()
