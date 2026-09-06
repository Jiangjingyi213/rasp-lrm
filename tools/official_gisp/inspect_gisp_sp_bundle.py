from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _to_float(value: Any) -> float:
    return float(torch.as_tensor(value).float().item())


def _summarize_masks(masks: dict[str, Any], suffix: str) -> dict[str, Any]:
    total_true = 0
    total = 0
    empty = []
    full = []
    per_layer = {}
    for key, value in sorted(masks.items(), key=lambda item: item[0]):
        if not key.endswith(suffix):
            continue
        mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").flatten()
        true_count = int(mask.sum().item())
        count = int(mask.numel())
        total_true += true_count
        total += count
        per_layer[key] = {"true": true_count, "total": count, "true_ratio": true_count / count if count else 0.0}
        if true_count == 0:
            empty.append(key)
        if true_count == count:
            full.append(key)
    return {
        "raw_true": total_true,
        "total": total,
        "raw_true_ratio": total_true / total if total else 0.0,
        "empty_masks": empty,
        "full_masks": full,
        "per_layer": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an official GISP sp_*.pth bundle without loading a model.")
    parser.add_argument("--sp-path", required=True)
    parser.add_argument("--show-layers", action="store_true")
    args = parser.parse_args()

    sp_path = Path(args.sp_path)
    bundle = torch.load(sp_path, map_location="cpu")
    if not isinstance(bundle, dict) or "actual_mask" not in bundle:
        raise ValueError(f"{sp_path} is not an official GISP bundle with actual_mask")

    masks = dict(bundle["actual_mask"])
    result = {
        "sp_path": str(sp_path),
        "avg_loss": _to_float(bundle.get("avg_loss", 0.0)),
        "block_wise_ratio": {
            str(key): _to_float(value) for key, value in dict(bundle.get("block_wise_ratio", {})).items()
        },
        "mlp_actual_mask": _summarize_masks(masks, ".mlp"),
        "attention_actual_mask": _summarize_masks(masks, ".self_attn"),
    }
    gqa_record = bundle.get("gqa_mask_record", {})
    if isinstance(gqa_record, dict) and gqa_record:
        gqa_true = 0
        gqa_total = 0
        for value in gqa_record.values():
            mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").flatten()
            gqa_true += int(mask.sum().item())
            gqa_total += int(mask.numel())
        result["gqa_mask_record"] = {
            "raw_true": gqa_true,
            "total": gqa_total,
            "raw_true_ratio": gqa_true / gqa_total if gqa_total else 0.0,
        }

    if not args.show_layers:
        result["mlp_actual_mask"].pop("per_layer", None)
        result["attention_actual_mask"].pop("per_layer", None)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
