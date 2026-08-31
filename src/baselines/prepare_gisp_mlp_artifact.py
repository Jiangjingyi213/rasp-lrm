from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.baselines.gisp_mlp_qwen3 import build_gisp_mlp_mask_artifact_qwen3
from src.models.load_model import load_model_bundle
from src.utils.io import read_jsonl, read_yaml, write_json


def _gisp_variant(cfg: dict[str, Any], method_name: str) -> dict[str, Any]:
    base = deepcopy(cfg.get("gisp_mlp", {}))
    variants = base.pop("variants", [])
    for variant in variants:
        if str(variant.get("method_name", variant.get("name", ""))) == method_name:
            merged = deepcopy(base)
            merged.update(deepcopy(variant))
            return merged
    raise ValueError(f"GISP-MLP method {method_name!r} was not found in config variants")


def _calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable GISP-MLP channel masks once.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    row = _gisp_variant(cfg, args.method)
    calibration_path = _calibration_path(row)
    if not Path(calibration_path).exists():
        raise FileNotFoundError(f"Missing GISP calibration file: {calibration_path}")
    calibration_rows = read_jsonl(calibration_path)
    sample_count = min(int(row.get("calibration_samples", 64)), len(calibration_rows))
    if sample_count <= 0:
        raise ValueError(f"No GISP calibration rows found at {calibration_path}")

    bundle = load_model_bundle(cfg["model"])
    payload = build_gisp_mlp_mask_artifact_qwen3(
        bundle.model,
        bundle.tokenizer,
        calibration_path=calibration_path,
        prompt_config=dict(cfg.get("prompt", {}).get("structured", {})),
        ratio=float(row.get("prune_ratio", row.get("ratio", 0.0))),
        iterations=int(row.get("iterations", 4)),
        calibration_samples=sample_count,
        max_input_tokens=int(row.get("calibration_max_input_tokens", 1024)),
        calibration_prompt_mode=str(row.get("calibration_prompt_mode", "structured_prompt")),
        calibration_text_field=str(row.get("calibration_text_field", "text")),
        score_normalization=str(row.get("score_normalization", "layer_mean")),
        layers=row.get("layers"),
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        target_matched_to_rasp_actual_mlp_pruning=(
            float(row["target_pruning_ratio"])
            if "target_pruning_ratio" in row
            else None
        ),
    )
    payload["method_name"] = args.method
    payload["config_path"] = args.config
    payload["calibration_path"] = calibration_path
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
