#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "generated_rasp_zero_online_conservative"
RUN_ROOT = "runs/03_rasp_zero/04_online_eval/rasp_zero_online_conservative_calibration"

BASE_CONFIGS = {
    "gsm8k": ROOT / "configs" / "exp_rasp_zero_runtime_router_gsm8k_smoke.yaml",
    "math500": ROOT / "configs" / "exp_rasp_zero_runtime_router_math500_smoke.yaml",
}

POLICIES = [
    {
        "name": "e64cap010_thr025_tgt010",
        "risk_threshold": 0.25,
        "target_average_ratio": 0.10,
        "early_tokens": 64,
        "early_max_ratio": 0.10,
    },
    {
        "name": "e96cap010_thr025_tgt010",
        "risk_threshold": 0.25,
        "target_average_ratio": 0.10,
        "early_tokens": 96,
        "early_max_ratio": 0.10,
    },
    {
        "name": "e96cap020_thr025_tgt010",
        "risk_threshold": 0.25,
        "target_average_ratio": 0.10,
        "early_tokens": 96,
        "early_max_ratio": 0.20,
    },
    {
        "name": "e96cap010_uncertain_thr025_tgt010",
        "risk_threshold": 0.25,
        "target_average_ratio": 0.10,
        "early_tokens": 96,
        "early_max_ratio": 0.10,
        "high_entropy_threshold": 1.0,
        "high_entropy_max_ratio": 0.10,
        "low_confidence_threshold": 0.70,
        "low_confidence_max_ratio": 0.10,
    },
]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset, base_path in BASE_CONFIGS.items():
        base = read_yaml(base_path)
        base["data"]["limit"] = 20
        for policy in POLICIES:
            cfg = copy.deepcopy(base)
            tag = f"{dataset}_{policy['name']}"
            run_dir = f"{RUN_ROOT}/{tag}"
            cfg["runtime_rasp"]["ratios"] = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
            cfg["runtime_rasp"]["default_max_ratio"] = 0.40
            for key, value in policy.items():
                if key != "name":
                    cfg["runtime_rasp"][key] = value
            cfg["paths"] = {
                "run_dir": run_dir,
                "trajectories": f"{run_dir}/01_trajectories.jsonl",
                "runtime_summary": f"{run_dir}/00_runtime_summary.json",
            }
            output = CONFIG_DIR / f"{tag}.yaml"
            write_yaml(output, cfg)
            manifest.append(
                {
                    "dataset": dataset,
                    "policy": policy["name"],
                    "config": str(output.relative_to(ROOT)),
                    "run_dir": run_dir,
                    **{key: value for key, value in policy.items() if key != "name"},
                }
            )
    with (CONFIG_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"generated {len(manifest)} conservative configs under {CONFIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
