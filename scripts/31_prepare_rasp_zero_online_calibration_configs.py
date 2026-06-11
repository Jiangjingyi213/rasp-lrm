#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "generated_rasp_zero_online_calibration"
RUN_ROOT = "runs/03_rasp_zero/04_online_eval/rasp_zero_online_calibration"

BASE_CONFIGS = {
    "gsm8k": ROOT / "configs" / "exp_rasp_zero_runtime_router_gsm8k_smoke.yaml",
    "math500": ROOT / "configs" / "exp_rasp_zero_runtime_router_math500_smoke.yaml",
}

POLICIES = [
    {"name": "thr025_tgt010", "risk_threshold": 0.25, "target_average_ratio": 0.10},
    {"name": "thr030_tgt010", "risk_threshold": 0.30, "target_average_ratio": 0.10},
    {"name": "thr030_tgt015", "risk_threshold": 0.30, "target_average_ratio": 0.15},
    {"name": "thr035_tgt015", "risk_threshold": 0.35, "target_average_ratio": 0.15},
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
            cfg["runtime_rasp"]["risk_threshold"] = policy["risk_threshold"]
            cfg["runtime_rasp"]["target_average_ratio"] = policy["target_average_ratio"]
            cfg["runtime_rasp"]["ratios"] = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
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
                    "risk_threshold": policy["risk_threshold"],
                    "target_average_ratio": policy["target_average_ratio"],
                    "config": str(output.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
    with (CONFIG_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"generated {len(manifest)} calibration configs under {CONFIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
