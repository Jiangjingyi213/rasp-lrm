from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.data.load_gsm8k import load_tasks
from src.main_generate import DEFAULT_STOP_STRINGS, truncate_completion
from src.metrics.answer_match import answer_match, extract_answer
from src.models.load_model import load_model_bundle
from src.rasp.budget_controller import ConfidenceThresholdController, FixedRatioController
from src.rasp.greedy_decode import greedy_decode_runtime
from src.rasp.metrics import summarize_runtime_rows
from src.rasp.mlp_runtime import apply_runtime_mlp_masking_qwen3
from src.utils.io import ensure_dir, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def _build_controller(config: dict):
    controller = str(config.get("controller", "fixed"))
    if controller == "fixed":
        return FixedRatioController(float(config.get("fixed_ratio", 0.0)))
    if controller == "confidence_threshold":
        return ConfidenceThresholdController(
            low_confidence=float(config.get("low_confidence", 0.50)),
            medium_confidence=float(config.get("medium_confidence", 0.75)),
            low_ratio=float(config.get("low_ratio", 0.0)),
            medium_ratio=float(config.get("medium_ratio", 0.05)),
            high_ratio=float(config.get("high_ratio", 0.10)),
        )
    raise ValueError(f"Unsupported runtime controller: {controller}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    set_seed(int(cfg.get("seed", 1)))
    runtime_cfg = cfg.get("runtime_rasp", {})
    run_dir = Path(cfg["paths"]["run_dir"])
    trajectories_path = Path(cfg["paths"].get("trajectories", run_dir / "01_trajectories.jsonl"))
    summary_path = Path(cfg["paths"].get("runtime_summary", run_dir / "00_runtime_summary.json"))
    ensure_dir(run_dir)

    bundle = load_model_bundle(cfg["model"])
    apply_runtime_mlp_masking_qwen3(
        bundle.model,
        ratios=runtime_cfg.get("ratios", [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]),
    )
    controller = _build_controller(runtime_cfg)
    tasks = load_tasks(cfg["data"])
    generation_cfg = cfg.get("generation", {})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for task in tqdm(tasks, desc="runtime-rasp"):
        prompt = build_prompt(task["question"], bundle.tokenizer, cfg.get("prompt", {}))
        runtime = greedy_decode_runtime(
            bundle.model,
            bundle.tokenizer,
            prompt,
            controller,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 512)),
            max_input_tokens=int(generation_cfg.get("max_input_tokens", 2048)),
            window_tokens=int(runtime_cfg.get("window_tokens", 16)),
        )
        completion = truncate_completion(runtime["completion"], generation_cfg.get("stop_strings", DEFAULT_STOP_STRINGS))
        rows.append(
            {
                **task,
                "prompt": prompt,
                "completion": completion,
                "prediction": extract_answer(completion),
                "correct": answer_match(completion, task.get("gold", "")),
                "runtime": runtime,
            }
        )
    write_jsonl(trajectories_path, rows)
    summary = {
        "method": "rasp_zero_runtime_v0",
        "backend": "logical_mask_v0",
        "controller": runtime_cfg.get("controller", "fixed"),
        "window_tokens": int(runtime_cfg.get("window_tokens", 16)),
        "supported_ratios": runtime_cfg.get("ratios", [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        **summarize_runtime_rows(rows),
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
