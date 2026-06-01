#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "generated_runtime_bank_l20"
RUN_ROOT = "runs/rasp_zero_runtime_bank_l20"
SHARDS = (
    ("gsm8k", 0, 0, "configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml"),
    ("gsm8k", 1, 10, "configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml"),
    ("math500", 0, 0, "configs/exp_rasp_zero_runtime_bank_math500_smoke.yaml"),
    ("math500", 1, 10, "configs/exp_rasp_zero_runtime_bank_math500_smoke.yaml"),
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def shard_config(dataset: str, shard: int, offset: int, source: str) -> dict:
    cfg = copy.deepcopy(load_yaml(ROOT / source))
    cfg["data"]["limit"] = 10
    cfg["data"]["offset"] = offset
    cfg["runtime_bank_validation"]["require_all_dense_correct"] = False
    run_dir = f"{RUN_ROOT}/{dataset}_s{shard}"
    cfg["paths"] = {
        "run_dir": run_dir,
        "trajectories": f"{run_dir}/01_trajectories.jsonl",
        "segments": f"{run_dir}/02_segments.jsonl",
        "counterfactuals": f"{run_dir}/03_counterfactuals.jsonl",
        "probe_dataset": f"{run_dir}/05_probe_dataset.jsonl",
        "probe_hidden_states": f"{run_dir}/05_probe_hidden_states.pt",
        "runtime_bank_validation": f"{run_dir}/07_runtime_bank_validation.json",
    }
    return cfg


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, shard, offset, source in SHARDS:
        output = CONFIG_DIR / f"{dataset}_s{shard}.yaml"
        write_yaml(output, shard_config(dataset, shard, offset, source))
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
