from __future__ import annotations

from pathlib import Path

import yaml


SOURCES = {
    "gsm8k": Path("configs/eval_dense_qwen3_gsm8k_official_thinking.yaml"),
    "math500": Path("configs/eval_dense_qwen3_math500_official_thinking.yaml"),
}


def main() -> None:
    output_dir = Path("configs/generated_official_thinking_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, source in SOURCES.items():
        with source.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["data"]["limit"] = 10
        config["generation"]["max_new_tokens"] = 8192
        run_dir = f"runs/02_baselines/eval_dense_qwen3_{dataset}_official_thinking_smoke"
        config["paths"]["run_dir"] = run_dir
        config["paths"]["trajectories"] = f"{run_dir}/01_trajectories.jsonl"
        destination = output_dir / f"{dataset}.yaml"
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(destination)


if __name__ == "__main__":
    main()

