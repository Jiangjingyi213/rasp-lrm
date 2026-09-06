from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"auto", ""}:
        return torch.float16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    obj: Any = model
    for part in "model.layers".split("."):
        if not hasattr(obj, part):
            raise ValueError("Could not find Qwen-style decoder layers at model.layers")
        obj = getattr(obj, part)
    return list(obj)


def _as_pruned_mask(value: Any, *, expected: int, key: str) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").flatten()
    if int(mask.numel()) != int(expected):
        raise ValueError(f"Mask {key!r} has {mask.numel()} entries, expected {expected}")
    return mask


def _zero_mlp(layer: nn.Module, pruned_mask: torch.Tensor) -> tuple[int, int]:
    mlp = getattr(layer, "mlp", None)
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj")

    width = int(mlp.gate_proj.weight.shape[0])
    if int(mlp.up_proj.weight.shape[0]) != width or int(mlp.down_proj.weight.shape[1]) != width:
        raise ValueError("Qwen3 MLP projections do not share one intermediate width")
    mask = _as_pruned_mask(pruned_mask, expected=width, key="mlp")
    idx = torch.where(mask)[0].to(mlp.gate_proj.weight.device)
    if idx.numel():
        mlp.gate_proj.weight.data.index_fill_(0, idx, 0)
        mlp.up_proj.weight.data.index_fill_(0, idx, 0)
        mlp.down_proj.weight.data.index_fill_(1, idx, 0)
        if getattr(mlp.gate_proj, "bias", None) is not None:
            mlp.gate_proj.bias.data.index_fill_(0, idx, 0)
        if getattr(mlp.up_proj, "bias", None) is not None:
            mlp.up_proj.bias.data.index_fill_(0, idx, 0)
    return int(idx.numel()), width


def _zero_attention(layer: nn.Module, pruned_mask: torch.Tensor) -> tuple[int, int]:
    attn = getattr(layer, "self_attn", None)
    if not all(hasattr(attn, name) for name in ("q_proj", "o_proj")):
        raise ValueError("Expected Qwen3 attention with q_proj/o_proj")

    q_out = int(attn.q_proj.weight.shape[0])
    o_in = int(attn.o_proj.weight.shape[1])
    num_heads = int(getattr(attn, "num_heads", 0) or getattr(getattr(attn, "config", None), "num_attention_heads", 0))
    if not num_heads:
        num_heads = int(pruned_mask.numel())
    if q_out % num_heads != 0 or o_in % num_heads != 0:
        raise ValueError(f"Cannot split attention projections into {num_heads} heads")
    mask = _as_pruned_mask(pruned_mask, expected=num_heads, key="self_attn")
    head_idx = torch.where(mask)[0]
    if head_idx.numel():
        q_head_dim = q_out // num_heads
        o_head_dim = o_in // num_heads
        q_rows = torch.cat(
            [torch.arange(int(i) * q_head_dim, (int(i) + 1) * q_head_dim) for i in head_idx]
        ).to(attn.q_proj.weight.device)
        o_cols = torch.cat(
            [torch.arange(int(i) * o_head_dim, (int(i) + 1) * o_head_dim) for i in head_idx]
        ).to(attn.o_proj.weight.device)
        attn.q_proj.weight.data.index_fill_(0, q_rows, 0)
        attn.o_proj.weight.data.index_fill_(1, o_cols, 0)
        if getattr(attn.q_proj, "bias", None) is not None:
            attn.q_proj.bias.data.index_fill_(0, q_rows, 0)
    return int(head_idx.numel()), num_heads


def _materialize_masks(model: nn.Module, masks: dict[str, Any]) -> dict[str, Any]:
    layers = _decoder_layers(model)
    mlp_pruned = 0
    mlp_total = 0
    attn_pruned = 0
    attn_total = 0
    per_layer: dict[str, dict[str, int]] = {}

    with torch.no_grad():
        for layer_id, layer in enumerate(layers):
            layer_summary: dict[str, int] = {}
            mlp_key = f"{layer_id}.mlp"
            if mlp_key in masks:
                pruned, total = _zero_mlp(layer, masks[mlp_key])
                mlp_pruned += pruned
                mlp_total += total
                layer_summary["mlp_pruned"] = pruned
                layer_summary["mlp_total"] = total
            attn_key = f"{layer_id}.self_attn"
            if attn_key in masks:
                pruned, total = _zero_attention(layer, masks[attn_key])
                attn_pruned += pruned
                attn_total += total
                layer_summary["attention_heads_pruned"] = pruned
                layer_summary["attention_heads_total"] = total
            per_layer[str(layer_id)] = layer_summary

    return {
        "mlp_pruned": mlp_pruned,
        "mlp_total": mlp_total,
        "mlp_pruned_ratio": mlp_pruned / mlp_total if mlp_total else 0.0,
        "attention_heads_pruned": attn_pruned,
        "attention_heads_total": attn_total,
        "attention_heads_pruned_ratio": attn_pruned / attn_total if attn_total else 0.0,
        "per_layer": per_layer,
    }


def _patch_missing_torch_dtensor_for_transformers_save() -> None:
    try:
        import torch.distributed.tensor as dist_tensor
    except Exception:
        return
    if hasattr(dist_tensor, "DTensor"):
        return

    class _RaspLrmMissingDTensor:
        pass

    dist_tensor.DTensor = _RaspLrmMissingDTensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an official GISP sp_*.pth mask bundle into a standard Qwen3 HF model by zeroing pruned structures."
    )
    parser.add_argument("--sp-path", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sp_path = Path(args.sp_path)
    output_dir = Path(args.output_dir)
    bundle = torch.load(sp_path, map_location="cpu")
    if not isinstance(bundle, dict) or "actual_mask" not in bundle:
        raise ValueError(f"{sp_path} is not an official GISP bundle with actual_mask")

    dtype = _dtype(args.torch_dtype)
    print(f"Loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    )
    class_name = type(model).__name__
    print(f"Loaded model class: {class_name}", flush=True)
    if "qwen3" in str(args.base_model).lower() and "Qwen3" not in class_name:
        raise RuntimeError(f"Refusing to materialize Qwen3 mask onto non-Qwen3 class: {class_name}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    summary = _materialize_masks(model, dict(bundle["actual_mask"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saving materialized model to: {output_dir}", flush=True)
    _patch_missing_torch_dtensor_for_transformers_save()
    model.save_pretrained(output_dir, safe_serialization=bool(args.safe_serialization))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=bool(args.trust_remote_code))
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "schema": "official_gisp_sp_materialized_qwen3_zero_mask_v1",
        "sp_path": str(sp_path),
        "base_model": str(args.base_model),
        "torch_dtype": str(args.torch_dtype),
        "mask_semantics": "official GISP actual_mask boolean True entries are zeroed as pruned structures",
        "materialization": "zeroed_weights_standard_hf_qwen3",
        "summary": summary,
        "block_wise_ratio": {
            key: float(torch.as_tensor(value).float().item())
            for key, value in dict(bundle.get("block_wise_ratio", {})).items()
        },
        "avg_loss": float(torch.as_tensor(bundle.get("avg_loss", 0.0)).float().item()),
    }
    (output_dir / "rasp_lrm_gisp_materialization.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("DONE materialize official GISP sp bundle", flush=True)


if __name__ == "__main__":
    main()
