from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.baselines.shortgpt_qwen3 import prepare_shortgpt_qwen3, summary_to_dict
from src.models.load_model import load_model_bundle
from src.utils.io import read_yaml, write_json


def _shortgpt_variant(cfg: dict[str, Any], method_name: str) -> dict[str, Any]:
    base = deepcopy(cfg.get("shortgpt", {}))
    variants = base.pop("variants", [])
    for variant in variants:
        if str(variant.get("method_name", variant.get("name", ""))) == method_name:
            merged = deepcopy(base)
            merged.update(deepcopy(variant))
            return merged
    raise ValueError(f"ShortGPT method {method_name!r} was not found in config variants")


def _calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ShortGPT pruned layer ids once.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="shortgpt_t30_matched")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    row = _shortgpt_variant(cfg, args.method)
    calibration_path = _calibration_path(row)
    if not Path(calibration_path).exists():
        raise FileNotFoundError(f"Missing ShortGPT calibration file: {calibration_path}")

    bundle = load_model_bundle(cfg["model"])
    summary, handles = prepare_shortgpt_qwen3(
        bundle.model,
        bundle.tokenizer,
        calibration_path=calibration_path,
        prompt_config=dict(cfg.get("prompt", {}).get("structured", {})),
        prune_ratio=float(row.get("prune_ratio", row.get("ratio", 0.0))),
        selection_method=str(row.get("selection_method", "block_influence")),
        calibration_samples=int(row.get("calibration_samples", 128)),
        max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        candidate_layers=row.get("candidate_layers"),
        protected_first_layers=int(row.get("protected_first_layers", 0)),
        protected_last_layers=int(row.get("protected_last_layers", 0)),
        pruned_layers=row.get("pruned_layers"),
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        target_matched_to_rasp_actual_mlp_pruning=(
            float(row["target_pruning_ratio"]) if "target_pruning_ratio" in row else None
        ),
    )
    for handle in handles:
        handle.remove()

    payload = summary_to_dict(summary)
    payload["method_name"] = args.method
    payload["config_path"] = args.config
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
