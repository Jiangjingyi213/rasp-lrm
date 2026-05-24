from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines.griffin_qwen3 import apply_griffin_qwen3


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def load_model_bundle(config: dict[str, Any]) -> ModelBundle:
    model_name = config["name_or_path"]
    dtype_name = config.get("dtype", "auto")
    dtype = "auto"
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=config.get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": config.get("trust_remote_code", True),
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    adapter = config.get("adapter")
    if adapter:
        if adapter != "griffin_qwen3":
            raise ValueError(f"Unsupported model adapter: {adapter}")
        model = apply_griffin_qwen3(
            model,
            density=float(config.get("griffin_density", 0.5)),
            selection_method=config.get("griffin_selection_method", "topk"),
            mode=config.get("griffin_mode", "gen"),
        )
    model.eval()
    device = next(model.parameters()).device
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)
