#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "generated_runtime_bank_formal"
RUN_ROOT = "runs/rasp_zero_runtime_bank_formal"
SOURCES = (
    ("gsm8k_train", "configs/exp_rasp_zero_runtime_bank_gsm8k_train.yaml"),
    ("math_train", "configs/exp_rasp_zero_runtime_bank_math_train.yaml"),
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def shard_config(source_name: str, source_path: str, shard: int, limit: int, offset: int) -> tuple[Path, str]:
    cfg = copy.deepcopy(load_yaml(ROOT / source_path))
    cfg["data"]["limit"] = limit
    cfg["data"]["offset"] = offset
    run_dir = f"{RUN_ROOT}/{source_name}_s{shard:02d}"
    cfg["paths"] = {
        "run_dir": run_dir,
        "trajectories": f"{run_dir}/01_trajectories.jsonl",
        "segments": f"{run_dir}/02_segments.jsonl",
        "counterfactuals": f"{run_dir}/03_counterfactuals.jsonl",
        "probe_dataset": f"{run_dir}/05_probe_dataset.jsonl",
        "probe_hidden_states": f"{run_dir}/05_probe_hidden_states.pt",
        "runtime_bank_validation": f"{run_dir}/07_runtime_bank_validation.json",
    }
    output = CONFIG_DIR / f"{source_name}_s{shard:02d}.yaml"
    write_yaml(output, cfg)
    return output, run_dir


def main() -> None:
    total_per_source = int(os.environ.get("RASP_BANK_LIMIT_PER_SOURCE", "500"))
    shard_size = int(os.environ.get("RASP_BANK_SHARD_SIZE", "50"))
    gpu_count = int(os.environ.get("RASP_BANK_GPU_COUNT", "4"))
    if total_per_source < 1 or shard_size < 1 or gpu_count < 1:
        raise ValueError("RASP bank limits must be positive")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    gpu = 0
    for source_name, source_path in SOURCES:
        for shard, offset in enumerate(range(0, total_per_source, shard_size)):
            limit = min(shard_size, total_per_source - offset)
            output, run_dir = shard_config(source_name, source_path, shard, limit, offset)
            manifest.append(
                {
                    "source": source_name,
                    "shard": shard,
                    "offset": offset,
                    "limit": limit,
                    "gpu_queue": gpu,
                    "config": str(output.relative_to(ROOT)),
                    "run_dir": run_dir,
                }
            )
            gpu = (gpu + 1) % gpu_count
    with (CONFIG_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    for physical_gpu in range(gpu_count):
        queue = CONFIG_DIR / f"gpu{physical_gpu}.list"
        with queue.open("w", encoding="utf-8") as f:
            for row in manifest:
                if row["gpu_queue"] == physical_gpu:
                    f.write(row["config"] + "\n")
    print(f"generated {len(manifest)} shards across {gpu_count} GPU queues under {CONFIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
