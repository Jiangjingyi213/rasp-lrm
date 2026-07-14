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
    SafeDynamicStageGriffinRuntime,
    StageMaskRuntime,
    StaticCoreResidualStageRuntime,
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
    if method["policy"] == "calibrated_stage_safe_dynamic_griffin":
        runtime = SafeDynamicStageGriffinRuntime(
            bank,
            stage_ratios=stage_ratios,
            runtime_weight=float(method.get("runtime_weight", 0.4)),
            prior_weight=float(method.get("prior_weight", 0.6)),
            warmup_tokens={
                stage: int(method.get("warmup_tokens", {}).get(stage, 0))
                for stage in STAGES
            },
            protected_core_ratios={
                stage: float(method.get("protected_core_ratios", {}).get(stage, 0.0))
                for stage in STAGES
            },
            refresh_intervals={
                stage: int(method.get("refresh_intervals", {}).get(stage, 0))
                for stage in STAGES
            },
            window_tokens={
                stage: int(method.get("window_tokens", {}).get(stage, 1))
                for stage in STAGES
            },
            bias_compensation=bool(method.get("bias_compensation", True)),
            prior_policy=str(method.get("prior_policy", "stage_specific")),
        )
        apply_adaptive_stage_griffin_qwen3(model, runtime)
        return runtime
    if method["policy"] == "calibrated_stage_static_core_residual_griffin":
        runtime = StaticCoreResidualStageRuntime(
            bank,
            stage_ratios=stage_ratios,
            base_policy=str(method.get("base_policy", "trajectory_global")),
            stage_prior_policy=str(method.get("stage_prior_policy", "stage_specific")),
            runtime_weight=float(method.get("runtime_weight", 0.3)),
            prior_weight=float(method.get("prior_weight", 0.7)),
            warmup_tokens={
                stage: int(method.get("warmup_tokens", {}).get(stage, 0))
                for stage in STAGES
            },
            static_core_ratios={
                stage: float(method.get("static_core_ratios", {}).get(stage, 1.0))
                for stage in STAGES
            },
            swap_ratios={
                stage: float(method.get("swap_ratios", {}).get(stage, 0.0))
                for stage in STAGES
            },
            refresh_intervals={
                stage: int(method.get("refresh_intervals", {}).get(stage, 0))
                for stage in STAGES
            },
            window_tokens={
                stage: int(method.get("window_tokens", {}).get(stage, 1))
                for stage in STAGES
            },
            bias_compensation=bool(method.get("bias_compensation", True)),
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
    actual = []
    actual_pruning_accounting = None
    runtime_backend = None
    runtime_alpha = None
    runtime_warmup_tokens = None
    runtime_score_mode = None
    runtime_runtime_weight = None
    runtime_prior_weight = None
    runtime_protected_core_ratios = None
    runtime_refresh_intervals = None
    runtime_window_tokens = None
    runtime_base_policy = None
    runtime_stage_prior_policy = None
    runtime_static_core_ratios = None
    runtime_swap_ratios = None
    runtime_actual_swapped_channels = None
    for row in rows:
        runtime_summary = row["runtime_stage_mask"]
        runtime_backend = runtime_backend or runtime_summary.get("backend")
        runtime_alpha = runtime_alpha if runtime_alpha is not None else runtime_summary.get("alpha")
        runtime_warmup_tokens = runtime_warmup_tokens or runtime_summary.get("warmup_tokens")
        runtime_score_mode = runtime_score_mode or runtime_summary.get("score_mode")
        runtime_runtime_weight = (
            runtime_runtime_weight
            if runtime_runtime_weight is not None
            else runtime_summary.get("runtime_weight")
        )
        runtime_prior_weight = (
            runtime_prior_weight
            if runtime_prior_weight is not None
            else runtime_summary.get("prior_weight")
        )
        runtime_protected_core_ratios = (
            runtime_protected_core_ratios or runtime_summary.get("protected_core_ratios")
        )
        runtime_refresh_intervals = (
            runtime_refresh_intervals or runtime_summary.get("refresh_intervals")
        )
        runtime_window_tokens = runtime_window_tokens or runtime_summary.get("window_tokens")
        runtime_base_policy = runtime_base_policy or runtime_summary.get("base_policy")
        runtime_stage_prior_policy = runtime_stage_prior_policy or runtime_summary.get("stage_prior_policy")
        runtime_static_core_ratios = runtime_static_core_ratios or runtime_summary.get("static_core_ratios")
        runtime_swap_ratios = runtime_swap_ratios or runtime_summary.get("swap_ratios")
        runtime_actual_swapped_channels = (
            runtime_actual_swapped_channels
            or runtime_summary.get("actual_swapped_channels_by_stage_layer")
        )
        stage_tokens.update(runtime_summary["tokens_by_stage"])
        dense_observation_tokens.update(runtime_summary.get("dense_observation_tokens_by_stage", {}))
        masked_tokens.update(runtime_summary.get("masked_tokens_by_stage", {}))
        mask_refresh_counts.update(runtime_summary.get("mask_refresh_count_by_stage", {}))
        if runtime_summary["fallback_reason"]:
            fallback[runtime_summary["fallback_reason"]] += 1
        theoretical.append(runtime_summary["theoretical_average_mlp_pruning_ratio"])
        actual.append(
            float(
                runtime_summary.get(
                    "actual_average_mlp_pruning_ratio",
                    runtime_summary["theoretical_average_mlp_pruning_ratio"],
                )
            )
        )
        actual_pruning_accounting = (
            actual_pruning_accounting
            or runtime_summary.get("actual_pruning_accounting")
            or "estimated_from_stage_ratios"
        )
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
        "actual_average_mlp_pruning_ratio": sum(actual) / len(actual) if actual else 0.0,
        "actual_pruning_accounting": actual_pruning_accounting or "estimated_from_stage_ratios",
    }
    if "target_pruning_ratio" in method:
        target = float(method["target_pruning_ratio"])
        actual_value = float(summary["actual_average_mlp_pruning_ratio"])
        summary["target_pruning_ratio"] = target
        summary["target_pruning_reached"] = bool(actual_value >= target)
        summary["target_pruning_gap"] = actual_value - target
        summary["target_pruning_status"] = (
            "passed" if actual_value >= target else "target_pruning_not_reached"
        )
    if "target_pruning_label" in method:
        summary["target_pruning_label"] = method["target_pruning_label"]
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
    if runtime_score_mode is not None:
        summary["adaptive_score_mode"] = runtime_score_mode
    if runtime_runtime_weight is not None:
        summary["adaptive_runtime_weight"] = runtime_runtime_weight
    if runtime_prior_weight is not None:
        summary["adaptive_prior_weight"] = runtime_prior_weight
    if runtime_protected_core_ratios is not None:
        summary["adaptive_protected_core_ratios"] = runtime_protected_core_ratios
    if runtime_refresh_intervals is not None:
        summary["adaptive_refresh_intervals"] = runtime_refresh_intervals
    if runtime_window_tokens is not None:
        summary["adaptive_window_tokens"] = runtime_window_tokens
    if runtime_base_policy is not None:
        summary["adaptive_base_policy"] = runtime_base_policy
    if runtime_stage_prior_policy is not None:
        summary["adaptive_stage_prior_policy"] = runtime_stage_prior_policy
    if runtime_static_core_ratios is not None:
        summary["adaptive_static_core_ratios"] = runtime_static_core_ratios
    if runtime_swap_ratios is not None:
        summary["adaptive_swap_ratios"] = runtime_swap_ratios
    if runtime_actual_swapped_channels is not None:
        summary["adaptive_actual_swapped_channels_by_stage_layer"] = runtime_actual_swapped_channels
    return rows, summary
