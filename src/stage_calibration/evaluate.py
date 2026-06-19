from __future__ import annotations

import random
from collections import Counter
from typing import Any

import torch
from tqdm import tqdm

from src.data.format_prompt import build_prompt, forced_assistant_prefix
from src.metrics.answer_match import answer_match, extract_answer

from .decode import decode_with_stage_masks
from .protocol import STAGES
from .runtime import StageMaskRuntime, apply_fixed_stage_masking_qwen3


def uniform_ratios(ratio: float) -> dict[str, float]:
    return {stage: float(ratio) for stage in STAGES}


def evaluate_method(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    bank: dict[str, Any],
    method: dict[str, Any],
    generation: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    runtime = StageMaskRuntime(
        bank,
        policy=str(method["policy"]),
        stage_ratios={stage: float(method["stage_ratios"].get(stage, 0.0)) for stage in STAGES},
        bias_compensation=bool(method.get("bias_compensation", True)),
    )
    apply_fixed_stage_masking_qwen3(model, runtime)
    rows = []
    for task in tqdm(tasks, desc=f"eval-{method['name']}"):
        prompt_cfg = dict(method.get("prompt", {}))
        prompt = build_prompt(task["question"], tokenizer, prompt_cfg)
        prefill = forced_assistant_prefix(prompt_cfg)
        result = decode_with_stage_masks(
            model,
            tokenizer,
            prompt,
            runtime,
            prefill=prefill,
            max_new_tokens=int(generation["max_new_tokens"]),
            max_input_tokens=int(generation.get("max_input_tokens", 4096)),
            temperature=float(generation.get("temperature", 0.6)),
            top_p=float(generation.get("top_p", 0.95)),
            top_k=int(generation.get("top_k", 20)),
        )
        rows.append(
            {
                **task,
                "method": method["name"],
                "prompt": prompt,
                **result,
                "prediction": extract_answer(result["completion"]),
                "correct": answer_match(result["completion"], str(task.get("gold", ""))),
            }
        )
    correct = sum(int(row["correct"]) for row in rows)
    stage_tokens = Counter()
    fallback = Counter()
    theoretical = []
    for row in rows:
        stage_tokens.update(row["runtime_stage_mask"]["tokens_by_stage"])
        if row["runtime_stage_mask"]["fallback_reason"]:
            fallback[row["runtime_stage_mask"]["fallback_reason"]] += 1
        theoretical.append(row["runtime_stage_mask"]["theoretical_average_mlp_pruning_ratio"])
    summary = {
        "method": method,
        "seed": seed,
        "problems": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "valid_stage_protocol_rate": (
            sum(int(row["stage_protocol"]["valid"]) for row in rows) / len(rows) if rows else None
        ),
        "fallback_rate": sum(fallback.values()) / len(rows) if rows else None,
        "fallback_reasons": dict(fallback),
        "truncation_rate": sum(int(row["truncated"]) for row in rows) / len(rows) if rows else None,
        "mean_generated_tokens": sum(row["generated_tokens"] for row in rows) / len(rows) if rows else None,
        "stage_tokens": dict(stage_tokens),
        "theoretical_average_mlp_pruning_ratio": sum(theoretical) / len(theoretical) if theoretical else 0.0,
    }
    return rows, summary
