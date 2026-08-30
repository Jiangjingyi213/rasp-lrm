from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.baselines.flap_mlp_qwen3 import build_flap_mlp_mask_artifact_qwen3
from src.data.format_prompt import build_prompt
from src.models.load_model import load_model_bundle
from src.utils.io import read_jsonl, read_yaml, write_json


def _flap_variant(cfg: dict[str, Any], method_name: str) -> dict[str, Any]:
    base = deepcopy(cfg.get("flap_mlp_official", {}))
    variants = base.pop("variants", [])
    for variant in variants:
        if str(variant.get("method_name", variant.get("name", ""))) == method_name:
            merged = deepcopy(base)
            merged.update(deepcopy(variant))
            return merged
    raise ValueError(f"FLAP-MLP method {method_name!r} was not found in config variants")


def _calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def _calibration_texts(tokenizer, cfg: dict[str, Any], row: dict[str, Any], sample_count: int) -> list[str]:
    calibration_rows = read_jsonl(_calibration_path(row))
    if not calibration_rows:
        raise ValueError(f"No FLAP calibration rows found at {_calibration_path(row)}")
    prompt_config = dict(cfg.get("prompt", {}).get("structured", {}))
    return [
        build_prompt(str(item["question"]), tokenizer, prompt_config)
        for item in calibration_rows[:sample_count]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable FLAP-MLP channel masks once.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    row = _flap_variant(cfg, args.method)
    calibration_path = _calibration_path(row)
    if not Path(calibration_path).exists():
        raise FileNotFoundError(f"Missing FLAP calibration file: {calibration_path}")
    if bool(row.get("bias_compensation", False)):
        raise ValueError("Reusable FLAP artifact path requires bias_compensation=false")
    if str(row.get("calibration_dataset", "mixed_calibration")) == "wikitext2":
        raise ValueError("prepare_flap_mlp_artifact currently expects calibration.jsonl rows")

    bundle = load_model_bundle(cfg["model"])
    sample_count = min(int(row.get("calibration_samples", 128)), len(read_jsonl(calibration_path)))
    texts = _calibration_texts(bundle.tokenizer, cfg, row, sample_count)
    payload = build_flap_mlp_mask_artifact_qwen3(
        bundle.model,
        bundle.tokenizer,
        calibration_texts=texts,
        ratio=float(row.get("prune_ratio", row.get("ratio", 0.0))),
        calibration_dataset=str(row.get("calibration_dataset", "mixed_calibration")),
        metric=str(row.get("metric", "WIFV")),
        structure=str(row.get("structure", "AL-AM")),
        calibration_samples=sample_count,
        max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        layers=row.get("layers"),
        bias_compensation=False,
    )
    payload["method_name"] = args.method
    payload["config_path"] = args.config
    payload["calibration_path"] = calibration_path
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
