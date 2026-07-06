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
from .runtime import (
    AdaptiveStageGriffinRuntime,
    StageMaskRuntime,
    apply_adaptive_stage_griffin_qwen3,
    apply_fixed_stage_masking_qwen3,
)


def uniform_ratios(ratio: float) -> dict[str, float]:
    return {stage: float(ratio) for stage in STAGES}


def _runtime_for_method(model, bank: dict[str, Any], method: dict[str, Any]):
    stage_ratios = {stage: float(method["stage_ratios"].get(stage, 0.0)) for stage in STAGES}
    if method["policy"] == "calibrated_stage_adaptive_griffin":
        runtime = AdaptiveStageGriffinRuntime(
            bank,
            stage_ratios=stage_ratios,
            alpha=float(method.get("alpha", 0.7)),
            warmup_tokens={
                stage: int(method.get("warmup_tokens", {}).get(stage, 0))
                for stage in STAGES
            },
            bias_compensation=bool(method.get("bias_compensation", True)),
            prior_policy=str(method.get("prior_policy", "stage_specific")),
        )
        apply_adaptive_stage_griffin_qwen3(model, runtime)
        return runtime
    runtime = StageMaskRuntime(
        bank,
        policy=str(method["policy"]),
        stage_ratios=stage_ratios,
        bias_compensation=bool(method.get("bias_compensation", True)),
    )
    apply_fixed_stage_masking_qwen3(model, runtime)
    return runtime


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
    runtime = _runtime_for_method(model, bank, method)
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
                "correct": answer_match(
                    result["completion"],
                    str(task.get("gold", "")),
                    answer_type=task.get("answer_type"),
                ),
            }
        )
    correct = sum(int(row["correct"]) for row in rows)
    stage_tokens = Counter()
    dense_observation_tokens = Counter()
    masked_tokens = Counter()
    mask_refresh_counts = Counter()
    fallback = Counter()
    theoretical = []
    runtime_backend = None
    runtime_alpha = None
    runtime_warmup_tokens = None
    for row in rows:
        runtime_summary = row["runtime_stage_mask"]
        runtime_backend = runtime_backend or runtime_summary.get("backend")
        runtime_alpha = runtime_alpha if runtime_alpha is not None else runtime_summary.get("alpha")
        runtime_warmup_tokens = runtime_warmup_tokens or runtime_summary.get("warmup_tokens")
        stage_tokens.update(runtime_summary["tokens_by_stage"])
        dense_observation_tokens.update(runtime_summary.get("dense_observation_tokens_by_stage", {}))
        masked_tokens.update(runtime_summary.get("masked_tokens_by_stage", {}))
        mask_refresh_counts.update(runtime_summary.get("mask_refresh_count_by_stage", {}))
        if runtime_summary["fallback_reason"]:
            fallback[runtime_summary["fallback_reason"]] += 1
        theoretical.append(runtime_summary["theoretical_average_mlp_pruning_ratio"])
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
    if runtime_backend:
        summary["runtime_backend"] = runtime_backend
    if dense_observation_tokens:
        summary["dense_observation_tokens_by_stage"] = dict(dense_observation_tokens)
    if masked_tokens:
        summary["masked_tokens_by_stage"] = dict(masked_tokens)
    if mask_refresh_counts:
        summary["mask_refresh_count_by_stage"] = dict(mask_refresh_counts)
    if runtime_alpha is not None:
        summary["adaptive_alpha"] = runtime_alpha
    if runtime_warmup_tokens is not None:
        summary["adaptive_warmup_tokens"] = runtime_warmup_tokens
    return rows, summary
