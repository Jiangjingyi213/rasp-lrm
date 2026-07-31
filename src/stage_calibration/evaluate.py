from __future__ import annotations

import random
from collections import Counter
from collections import defaultdict
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.data.format_prompt import build_prompt, forced_assistant_prefix
from src.metrics.answer_match import answer_match, extract_answer
from src.baselines.flap_mlp_qwen3 import (
    apply_flap_mlp_pruning_qwen3,
    summary_to_dict as flap_summary_to_dict,
)
from src.baselines.llm_pruner_mlp_qwen3 import (
    apply_llm_pruner_mlp_pruning_qwen3,
    summary_to_dict as llm_pruner_mlp_summary_to_dict,
)
from src.baselines.sparsegpt_official_qwen3 import (
    apply_sparsegpt_official_qwen3_artifact,
    summary_to_dict as sparsegpt_summary_to_dict,
)
from src.baselines.shortgpt_qwen3 import (
    prepare_shortgpt_qwen3,
    summary_to_dict as shortgpt_summary_to_dict,
)
from src.baselines.wanda_official_qwen3 import apply_wanda_official_qwen3, summary_to_dict

from .decode import decode_with_stage_masks
from .protocol import STAGES
from .artifacts import file_sha256, stable_hash
from .runtime import (
    AdaptiveStageGriffinRuntime,
    AlwaysOnStaticMaskRuntime,
    DenseStageRuntime,
    GriffinPromptRuntime,
    SafeDynamicStageGriffinRuntime,
    StageRiskAdaptiveRuntime,
    StaticLayerPruningRuntime,
    StaticMlpChannelPruningRuntime,
    StageMaskRuntime,
    StaticWeightPruningRuntime,
    StaticCoreResidualStageRuntime,
    apply_adaptive_stage_griffin,
    apply_fixed_stage_masking,
    apply_griffin_prompt,
)
from src.utils.io import read_jsonl


def uniform_ratios(ratio: float) -> dict[str, float]:
    return {stage: float(ratio) for stage in STAGES}


def method_requires_mask_bank(method: dict[str, Any]) -> bool:
    if all(float(value) <= 0.0 for value in method.get("stage_ratios", {}).values()):
        return False
    return method.get("policy") not in {
        "griffin_prompt",
        "wanda_official",
        "sparsegpt_official",
        "shortgpt",
        "limits_layer_pruning",
        "flap_mlp_official",
        "llm_pruner_mlp_static_width",
    }


def _flap_calibration_texts(tokenizer, method: dict[str, Any], sample_count: int) -> list[str]:
    source = str(method.get("calibration_dataset", "mixed_calibration"))
    max_input_tokens = int(method.get("calibration_max_input_tokens", 2048))
    if source == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join(str(row.get("text", "")).strip() for row in dataset if str(row.get("text", "")).strip())
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if int(token_ids.numel()) <= max_input_tokens:
            return [text]
        rng = random.Random(int(method.get("calibration_seed", 3)))
        chunks = []
        upper = int(token_ids.numel()) - max_input_tokens - 1
        for _ in range(sample_count):
            start = rng.randint(0, upper)
            chunk_ids = token_ids[start : start + max_input_tokens]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks

    calibration_rows = read_jsonl(str(method["calibration_path"]))
    if not calibration_rows:
        raise ValueError(f"No FLAP calibration rows found at {method['calibration_path']}")
    prompt_config = dict(method.get("prompt", {}))
    return [
        build_prompt(str(row["question"]), tokenizer, prompt_config)
        for row in calibration_rows[:sample_count]
    ]


def _runtime_for_method(model, bank: dict[str, Any] | None, method: dict[str, Any]):
    if method["policy"] == "griffin_prompt":
        runtime = GriffinPromptRuntime(
            prune_ratio=float(method.get("prune_ratio", method["stage_ratios"].get("setup", 0.0))),
            selection_method=str(method.get("selection_method", "topk")),
        )
        apply_griffin_prompt(model, runtime)
        return runtime
    if method["policy"] == "wanda_official":
        wanda_summary = apply_wanda_official_qwen3(
            model,
            method["tokenizer"],
            calibration_path=str(method["calibration_path"]),
            prompt_config=dict(method.get("prompt", {})),
            sparsity_ratio=float(method.get("sparsity_ratio", method["stage_ratios"].get("setup", 0.0))),
            calibration_samples=int(method.get("calibration_samples", 128)),
            max_input_tokens=int(method.get("calibration_max_input_tokens", 2048)),
            calibration_prompt_mode=str(method.get("calibration_prompt_mode", "structured_prompt")),
            calibration_text_field=str(method.get("calibration_text_field", "text")),
            calibration_seed=(
                int(method["calibration_seed"])
                if method.get("calibration_seed") is not None
                else None
            ),
            target_modules=method.get("target_modules"),
            matched_rasp_reference=str(method.get("matched_rasp_reference", "")),
            target_matched_to_rasp_actual_mlp_pruning=(
                float(method["target_pruning_ratio"])
                if "target_pruning_ratio" in method
                else None
            ),
        )
        summary = summary_to_dict(wanda_summary)
        return StaticWeightPruningRuntime(
            policy="wanda_official",
            backend="wanda_official_weight_mask_v1",
            baseline_type="official_style_wanda",
            pruning_granularity="weight_unstructured",
            weight_sparsity_overall=float(summary["weight_sparsity_overall"]),
            extra_summary={
                "wanda_official_summary": summary,
                "wanda_sparsity_ratio": float(summary["sparsity_ratio"]),
                "wanda_weight_sparsity": float(summary["weight_sparsity_overall"]),
                "wanda_calibration_samples": int(summary["calibration_samples"]),
                "wanda_calibration_source": str(summary["calibration_path"]),
                "wanda_calibration_prompt_mode": str(summary["calibration_prompt_mode"]),
                "wanda_calibration_text_field": str(summary["calibration_text_field"]),
                "wanda_calibration_seed": summary["calibration_seed"],
                "wanda_target_modules": list(summary["target_modules"]),
                "weight_sparsity_by_module": dict(summary["weight_sparsity_by_module"]),
                "matched_rasp_reference": str(summary["matched_rasp_reference"]),
                "target_matched_to_rasp_actual_mlp_pruning": summary[
                    "target_matched_to_rasp_actual_mlp_pruning"
                ],
                "real_speedup_claimed": False,
            },
        )
    if method["policy"] == "sparsegpt_official":
        sparsegpt_summary = apply_sparsegpt_official_qwen3_artifact(
            model,
            artifact_dir=str(method["artifact_path"]),
        )
        summary = sparsegpt_summary_to_dict(sparsegpt_summary)
        return StaticWeightPruningRuntime(
            policy="sparsegpt_official",
            backend="sparsegpt_official_weight_mask_v1",
            baseline_type="official_style_sparsegpt",
            pruning_granularity="weight_unstructured",
            weight_sparsity_overall=float(summary["weight_sparsity_overall"]),
            extra_summary={
                "sparsegpt_official_summary": summary,
                "sparsegpt_sparsity_ratio": float(summary["sparsity_ratio"]),
                "sparsegpt_weight_sparsity": float(summary["weight_sparsity_overall"]),
                "sparsegpt_calibration_samples": int(summary["calibration_samples"]),
                "sparsegpt_calibration_source": str(summary["calibration_path"]),
                "sparsegpt_target_modules": list(summary["target_modules"]),
                "sparsegpt_artifact_path": str(summary["artifact_path"]),
                "sparsegpt_artifact_hash": str(summary["artifact_hash"]),
                "sparsegpt_blocksize": int(summary["blocksize"]),
                "sparsegpt_percdamp": float(summary["percdamp"]),
                "weight_sparsity_by_module": dict(summary["weight_sparsity_by_module"]),
                "matched_rasp_reference": str(summary["matched_rasp_reference"]),
                "target_matched_to_rasp_actual_mlp_pruning": summary[
                    "target_matched_to_rasp_actual_mlp_pruning"
                ],
                "real_speedup_claimed": False,
            },
        )
    if method["policy"] == "flap_mlp_official":
        source = str(method.get("calibration_dataset", "mixed_calibration"))
        if source == "wikitext2":
            sample_count = int(method.get("calibration_samples", 128))
        else:
            calibration_rows = read_jsonl(str(method["calibration_path"]))
            sample_count = min(int(method.get("calibration_samples", 128)), len(calibration_rows))
        if sample_count <= 0:
            raise ValueError(f"No FLAP calibration rows found at {method['calibration_path']}")
        calibration_texts = _flap_calibration_texts(method["tokenizer"], method, sample_count)
        flap_summary = apply_flap_mlp_pruning_qwen3(
            model,
            method["tokenizer"],
            calibration_texts=calibration_texts,
            ratio=float(method.get("prune_ratio", method["stage_ratios"].get("setup", 0.0))),
            calibration_dataset=str(method.get("calibration_dataset", "mixed_calibration")),
            metric=str(method.get("metric", "WIFV")),
            structure=str(method.get("structure", "AL-AM")),
            calibration_samples=sample_count,
            max_input_tokens=int(method.get("calibration_max_input_tokens", 2048)),
            layers=method.get("layers"),
            bias_compensation=bool(method.get("bias_compensation", False)),
        )
        summary = flap_summary_to_dict(flap_summary)
        total_pruned = sum(int(value) for value in summary["pruned_channels_per_layer"].values())
        total_channels = int(summary["original_intermediate_size"]) * len(summary["pruned_layers"])
        actual_ratio = total_pruned / total_channels if total_channels else 0.0
        return StaticMlpChannelPruningRuntime(
            policy="flap_mlp_official",
            backend="flap_mlp_qwen3_physical_channel_pruning_v1",
            baseline_type="official_style_flap_mlp_qwen3_port",
            pruning_granularity="mlp_channel_structured",
            mlp_channel_pruning_ratio=actual_ratio,
            extra_summary={
                "flap_mlp_official_summary": summary,
                "flap_metric": str(summary["metric"]),
                "flap_structure": str(summary["structure"]),
                "flap_prune_ratio": float(summary["ratio"]),
                "flap_calibration_dataset": str(summary["calibration_dataset"]),
                "flap_calibration_samples": int(summary["calibration_samples"]),
                "flap_calibration_source": str(method["calibration_path"]),
                "flap_bias_compensation": bool(summary["bias_compensation"]),
                "flap_physical_pruning": bool(summary["physical_pruning"]),
                "flap_target": str(summary["target"]),
                "flap_kept_channels_per_layer": dict(summary["kept_channels_per_layer"]),
                "flap_pruned_channels_per_layer": dict(summary["pruned_channels_per_layer"]),
                "flap_actual_mlp_channel_pruning_ratio": actual_ratio,
                "matched_rasp_reference": str(method.get("matched_rasp_reference", "")),
                "target_matched_to_rasp_actual_mlp_pruning": (
                    float(method["target_pruning_ratio"])
                    if "target_pruning_ratio" in method
                    else None
                ),
                "real_speedup_claimed": False,
            },
        )
    if method["policy"] == "llm_pruner_mlp_static_width":
        llm_pruner_summary = apply_llm_pruner_mlp_pruning_qwen3(
            model,
            ratio=float(method.get("prune_ratio", method["stage_ratios"].get("setup", 0.0))),
            importance=str(method.get("importance", "l2")),
            structure=str(method.get("structure", "UL-UM")),
            layers=method.get("layers"),
            physical_pruning=bool(method.get("physical_pruning", True)),
        )
        summary = llm_pruner_mlp_summary_to_dict(llm_pruner_summary)
        total_pruned = sum(int(value) for value in summary["pruned_channels_per_layer"].values())
        total_channels = int(summary["original_intermediate_size"]) * len(summary["pruned_layers"])
        actual_ratio = total_pruned / total_channels if total_channels else 0.0
        return StaticMlpChannelPruningRuntime(
            policy="llm_pruner_mlp_static_width",
            backend="llm_pruner_mlp_qwen3_physical_channel_pruning_v1"
            if bool(summary["physical_pruning"])
            else "llm_pruner_mlp_qwen3_logical_channel_mask_v1",
            baseline_type="llm_pruner_style_static_width_qwen3_mlp_no_recovery",
            pruning_granularity="mlp_channel_structured",
            mlp_channel_pruning_ratio=actual_ratio,
            extra_summary={
                "llm_pruner_mlp_summary": summary,
                "llm_pruner_importance": str(summary["importance"]),
                "llm_pruner_structure": str(summary["structure"]),
                "llm_pruner_prune_ratio": float(summary["ratio"]),
                "llm_pruner_physical_pruning": bool(summary["physical_pruning"]),
                "llm_pruner_target": str(summary["target"]),
                "llm_pruner_kept_channels_per_layer": dict(summary["kept_channels_per_layer"]),
                "llm_pruner_pruned_channels_per_layer": dict(summary["pruned_channels_per_layer"]),
                "llm_pruner_actual_mlp_channel_pruning_ratio": actual_ratio,
                "matched_rasp_reference": str(method.get("matched_rasp_reference", "")),
                "target_matched_to_rasp_actual_mlp_pruning": (
                    float(method["target_pruning_ratio"])
                    if "target_pruning_ratio" in method
                    else None
                ),
                "real_speedup_claimed": False,
            },
        )
    if method["policy"] in {"shortgpt", "limits_layer_pruning"}:
        shortgpt_summary, handles = prepare_shortgpt_qwen3(
            model,
            method["tokenizer"],
            calibration_path=str(method["calibration_path"]),
            prompt_config=dict(method.get("prompt", {})),
            prune_ratio=float(method.get("prune_ratio", method["stage_ratios"].get("setup", 0.0))),
            selection_method=str(method.get("selection_method", "block_influence")),
            calibration_samples=int(method.get("calibration_samples", 128)),
            max_input_tokens=int(method.get("calibration_max_input_tokens", 2048)),
            candidate_layers=method.get("candidate_layers"),
            protected_first_layers=int(method.get("protected_first_layers", 0)),
            protected_last_layers=int(method.get("protected_last_layers", 0)),
            pruned_layers=method.get("pruned_layers"),
            matched_rasp_reference=str(method.get("matched_rasp_reference", "")),
            target_matched_to_rasp_actual_mlp_pruning=(
                float(method["target_pruning_ratio"])
                if "target_pruning_ratio" in method
                else None
            ),
        )
        summary = shortgpt_summary_to_dict(shortgpt_summary)
        policy = str(method["policy"])
        baseline_type = str(method.get("baseline_type", summary["baseline_type"]))
        backend = (
            "limits_reverse_layer_skip_logical_v1"
            if policy == "limits_layer_pruning"
            else "shortgpt_layer_skip_logical_v1"
        )
        return StaticLayerPruningRuntime(
            policy=policy,
            backend=backend,
            baseline_type=baseline_type,
            pruning_granularity="decoder_layer_logical_skip",
            total_layers=int(summary["total_layers"]),
            pruned_layers=list(summary["pruned_layers"]),
            handles=handles,
            extra_summary={
                "shortgpt_summary": summary,
                "shortgpt_prune_ratio": float(summary["prune_ratio"]),
                "shortgpt_layer_pruning_ratio": (
                    len(summary["pruned_layers"]) / int(summary["total_layers"])
                    if int(summary["total_layers"])
                    else 0.0
                ),
                "shortgpt_calibration_samples": int(summary["calibration_samples"]),
                "shortgpt_calibration_source": str(summary["calibration_path"]),
                "shortgpt_candidate_layers": list(summary["candidate_layers"]),
                "shortgpt_pruned_layers": list(summary["pruned_layers"]),
                "shortgpt_block_influence_by_layer": dict(summary["block_influence_by_layer"]),
                "layer_pruning_selection_method": str(summary["selection_method"]),
                "matched_rasp_reference": str(summary["matched_rasp_reference"]),
                "target_matched_to_rasp_actual_mlp_pruning": summary[
                    "target_matched_to_rasp_actual_mlp_pruning"
                ],
                "real_speedup_claimed": False,
            },
        )
    if all(float(value) <= 0.0 for value in method.get("stage_ratios", {}).values()):
        return DenseStageRuntime(policy=str(method.get("policy", "dense")))
    if bank is None:
        raise ValueError(f"Method {method['name']} with policy {method['policy']} requires a mask bank")
    stage_ratios = {stage: float(method["stage_ratios"].get(stage, 0.0)) for stage in STAGES}
    if str(method["policy"]).endswith("_always_on"):
        base_policy = str(method["policy"])[: -len("_always_on")]
        runtime = AlwaysOnStaticMaskRuntime(
            bank,
            policy=base_policy,
            stage_ratios=stage_ratios,
            bias_compensation=bool(method.get("bias_compensation", True)),
        )
        apply_fixed_stage_masking(model, runtime)
        return runtime
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
            fallback_behavior=str(method.get("fallback_behavior", "dense_after_error")),
        )
        apply_adaptive_stage_griffin(model, runtime)
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
            fallback_behavior=str(method.get("fallback_behavior", "dense_after_error")),
        )
        apply_adaptive_stage_griffin(model, runtime)
        return runtime
    if method["policy"] == "stage_risk_adaptive":
        controller_cfg = dict(method.get("stage_risk_controller", {}))
        runtime = StageRiskAdaptiveRuntime(
            bank,
            stage_ratios=stage_ratios,
            controller_checkpoint_path=str(controller_cfg["checkpoint_path"]),
            action_ratios=[float(value) for value in controller_cfg["action_ratios"]],
            risk_thresholds={stage: float(controller_cfg["risk_thresholds"][stage]) for stage in STAGES},
            stage_ratio_caps={stage: float(controller_cfg["stage_ratio_caps"][stage]) for stage in STAGES},
            min_warmup_tokens={
                stage: int(controller_cfg.get("min_warmup_tokens", {}).get(stage, method.get("warmup_tokens", {}).get(stage, 0)))
                for stage in STAGES
            },
            decision_window_tokens=int(controller_cfg.get("decision_window_tokens", 64)),
            target_actual_pruning=float(controller_cfg.get("target_actual_pruning", 0.34)),
            max_mask_swap_fraction=float(controller_cfg.get("max_mask_swap_fraction", 0.05)),
            score_mode=str(controller_cfg.get("score_mode", "current_safe")),
            fallback_behavior=str(
                controller_cfg.get("fallback_behavior", method.get("fallback_behavior", "dense_after_error"))
            ),
            runtime_weight=float(method.get("runtime_weight", 0.4)),
            prior_weight=float(method.get("prior_weight", 0.6)),
            warmup_tokens={stage: int(method.get("warmup_tokens", {}).get(stage, 0)) for stage in STAGES},
            protected_core_ratios={stage: float(method.get("protected_core_ratios", {}).get(stage, 0.0)) for stage in STAGES},
            refresh_intervals={stage: int(method.get("refresh_intervals", {}).get(stage, 0)) for stage in STAGES},
            window_tokens={stage: int(method.get("window_tokens", {}).get(stage, 1)) for stage in STAGES},
            bias_compensation=bool(method.get("bias_compensation", True)),
            prior_policy=str(method.get("prior_policy", "stage_specific")),
        )
        apply_adaptive_stage_griffin(model, runtime)
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
        apply_adaptive_stage_griffin(model, runtime)
        return runtime
    runtime = StageMaskRuntime(
        bank,
        policy=str(method["policy"]),
        stage_ratios=stage_ratios,
        bias_compensation=bool(method.get("bias_compensation", True)),
        fallback_behavior=str(method.get("fallback_behavior", "dense_after_error")),
    )
    apply_fixed_stage_masking(model, runtime)
    return runtime


def evaluate_method(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    bank: dict[str, Any] | None,
    method: dict[str, Any],
    generation: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    method = dict(method)
    if method.get("policy") in {
        "wanda_official",
        "shortgpt",
        "limits_layer_pruning",
        "flap_mlp_official",
    }:
        method["tokenizer"] = tokenizer
    runtime = _runtime_for_method(model, bank, method)
    method.pop("tokenizer", None)
    rows = []
    try:
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
    finally:
        close = getattr(runtime, "close", None)
        if close is not None:
            close()
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
    runtime_baseline_type = None
    runtime_selection_method = None
    runtime_prune_ratio = None
    runtime_density = None
    prompt_dense_tokens = 0
    decode_masked_tokens = 0
    keep_ratios_by_layer = None
    wanda_weight_sparsity = None
    wanda_sparsity_ratio = None
    wanda_calibration_samples = None
    wanda_calibration_source = None
    wanda_calibration_prompt_mode = None
    wanda_calibration_text_field = None
    wanda_calibration_seed = None
    wanda_target_modules = None
    sparsegpt_weight_sparsity = None
    sparsegpt_sparsity_ratio = None
    sparsegpt_calibration_samples = None
    sparsegpt_calibration_source = None
    sparsegpt_target_modules = None
    sparsegpt_artifact_path = None
    sparsegpt_artifact_hash = None
    sparsegpt_blocksize = None
    sparsegpt_percdamp = None
    shortgpt_calibration_samples = None
    shortgpt_calibration_source = None
    shortgpt_candidate_layers = None
    shortgpt_pruned_layers = None
    shortgpt_block_influence_by_layer = None
    shortgpt_layer_pruning_ratio = None
    flap_metric = None
    flap_structure = None
    flap_prune_ratio = None
    flap_calibration_dataset = None
    flap_calibration_samples = None
    flap_calibration_source = None
    flap_bias_compensation = None
    flap_physical_pruning = None
    flap_target = None
    flap_kept_channels_per_layer = None
    flap_pruned_channels_per_layer = None
    flap_actual_mlp_channel_pruning_ratio = None
    llm_pruner_importance = None
    llm_pruner_structure = None
    llm_pruner_prune_ratio = None
    llm_pruner_physical_pruning = None
    llm_pruner_target = None
    llm_pruner_kept_channels_per_layer = None
    llm_pruner_pruned_channels_per_layer = None
    llm_pruner_actual_mlp_channel_pruning_ratio = None
    weight_sparsity_by_module = None
    matched_rasp_reference = None
    target_matched_to_rasp_actual_mlp_pruning = None
    pruning_granularity = None
    stage_risk_actions = Counter()
    stage_risk_ratio_tokens = Counter()
    stage_risk_risks: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stage_risk_checkpoints = set()
    stage_risk_mask_swaps = 0
    stage_risk_mask_swap_candidates = 0
    stage_risk_mask_jaccards = []
    for row in rows:
        runtime_summary = row["runtime_stage_mask"]
        runtime_backend = runtime_backend or runtime_summary.get("backend")
        runtime_baseline_type = runtime_baseline_type or runtime_summary.get("baseline_type")
        runtime_selection_method = runtime_selection_method or runtime_summary.get("selection_method")
        runtime_prune_ratio = (
            runtime_prune_ratio if runtime_prune_ratio is not None else runtime_summary.get("prune_ratio")
        )
        runtime_density = runtime_density if runtime_density is not None else runtime_summary.get("density")
        prompt_dense_tokens += int(runtime_summary.get("prompt_dense_tokens", 0))
        decode_masked_tokens += int(runtime_summary.get("decode_masked_tokens", 0))
        keep_ratios_by_layer = keep_ratios_by_layer or runtime_summary.get("keep_ratios_by_layer")
        wanda_weight_sparsity = (
            wanda_weight_sparsity
            if wanda_weight_sparsity is not None
            else runtime_summary.get("wanda_weight_sparsity")
        )
        wanda_sparsity_ratio = (
            wanda_sparsity_ratio
            if wanda_sparsity_ratio is not None
            else runtime_summary.get("wanda_sparsity_ratio")
        )
        wanda_calibration_samples = (
            wanda_calibration_samples
            if wanda_calibration_samples is not None
            else runtime_summary.get("wanda_calibration_samples")
        )
        wanda_calibration_source = (
            wanda_calibration_source
            or runtime_summary.get("wanda_calibration_source")
        )
        wanda_calibration_prompt_mode = (
            wanda_calibration_prompt_mode
            or runtime_summary.get("wanda_calibration_prompt_mode")
        )
        wanda_calibration_text_field = (
            wanda_calibration_text_field
            or runtime_summary.get("wanda_calibration_text_field")
        )
        wanda_calibration_seed = (
            wanda_calibration_seed
            if wanda_calibration_seed is not None
            else runtime_summary.get("wanda_calibration_seed")
        )
        wanda_target_modules = wanda_target_modules or runtime_summary.get("wanda_target_modules")
        sparsegpt_weight_sparsity = (
            sparsegpt_weight_sparsity
            if sparsegpt_weight_sparsity is not None
            else runtime_summary.get("sparsegpt_weight_sparsity")
        )
        sparsegpt_sparsity_ratio = (
            sparsegpt_sparsity_ratio
            if sparsegpt_sparsity_ratio is not None
            else runtime_summary.get("sparsegpt_sparsity_ratio")
        )
        sparsegpt_calibration_samples = (
            sparsegpt_calibration_samples
            if sparsegpt_calibration_samples is not None
            else runtime_summary.get("sparsegpt_calibration_samples")
        )
        sparsegpt_calibration_source = (
            sparsegpt_calibration_source
            or runtime_summary.get("sparsegpt_calibration_source")
        )
        sparsegpt_target_modules = sparsegpt_target_modules or runtime_summary.get(
            "sparsegpt_target_modules"
        )
        sparsegpt_artifact_path = sparsegpt_artifact_path or runtime_summary.get("sparsegpt_artifact_path")
        sparsegpt_artifact_hash = sparsegpt_artifact_hash or runtime_summary.get("sparsegpt_artifact_hash")
        sparsegpt_blocksize = (
            sparsegpt_blocksize
            if sparsegpt_blocksize is not None
            else runtime_summary.get("sparsegpt_blocksize")
        )
        sparsegpt_percdamp = (
            sparsegpt_percdamp
            if sparsegpt_percdamp is not None
            else runtime_summary.get("sparsegpt_percdamp")
        )
        shortgpt_calibration_samples = (
            shortgpt_calibration_samples
            if shortgpt_calibration_samples is not None
            else runtime_summary.get("shortgpt_calibration_samples")
        )
        shortgpt_calibration_source = (
            shortgpt_calibration_source
            or runtime_summary.get("shortgpt_calibration_source")
        )
        shortgpt_candidate_layers = (
            shortgpt_candidate_layers or runtime_summary.get("shortgpt_candidate_layers")
        )
        shortgpt_pruned_layers = (
            shortgpt_pruned_layers or runtime_summary.get("shortgpt_pruned_layers")
        )
        shortgpt_block_influence_by_layer = (
            shortgpt_block_influence_by_layer
            or runtime_summary.get("shortgpt_block_influence_by_layer")
        )
        shortgpt_layer_pruning_ratio = (
            shortgpt_layer_pruning_ratio
            if shortgpt_layer_pruning_ratio is not None
            else runtime_summary.get("shortgpt_layer_pruning_ratio")
        )
        flap_metric = flap_metric or runtime_summary.get("flap_metric")
        flap_structure = flap_structure or runtime_summary.get("flap_structure")
        flap_prune_ratio = (
            flap_prune_ratio
            if flap_prune_ratio is not None
            else runtime_summary.get("flap_prune_ratio")
        )
        flap_calibration_dataset = (
            flap_calibration_dataset or runtime_summary.get("flap_calibration_dataset")
        )
        flap_calibration_samples = (
            flap_calibration_samples
            if flap_calibration_samples is not None
            else runtime_summary.get("flap_calibration_samples")
        )
        flap_calibration_source = (
            flap_calibration_source or runtime_summary.get("flap_calibration_source")
        )
        flap_bias_compensation = (
            flap_bias_compensation
            if flap_bias_compensation is not None
            else runtime_summary.get("flap_bias_compensation")
        )
        flap_physical_pruning = (
            flap_physical_pruning
            if flap_physical_pruning is not None
            else runtime_summary.get("flap_physical_pruning")
        )
        flap_target = flap_target or runtime_summary.get("flap_target")
        flap_kept_channels_per_layer = (
            flap_kept_channels_per_layer or runtime_summary.get("flap_kept_channels_per_layer")
        )
        flap_pruned_channels_per_layer = (
            flap_pruned_channels_per_layer or runtime_summary.get("flap_pruned_channels_per_layer")
        )
        flap_actual_mlp_channel_pruning_ratio = (
            flap_actual_mlp_channel_pruning_ratio
            if flap_actual_mlp_channel_pruning_ratio is not None
            else runtime_summary.get("flap_actual_mlp_channel_pruning_ratio")
        )
        llm_pruner_importance = llm_pruner_importance or runtime_summary.get("llm_pruner_importance")
        llm_pruner_structure = llm_pruner_structure or runtime_summary.get("llm_pruner_structure")
        llm_pruner_prune_ratio = (
            llm_pruner_prune_ratio
            if llm_pruner_prune_ratio is not None
            else runtime_summary.get("llm_pruner_prune_ratio")
        )
        llm_pruner_physical_pruning = (
            llm_pruner_physical_pruning
            if llm_pruner_physical_pruning is not None
            else runtime_summary.get("llm_pruner_physical_pruning")
        )
        llm_pruner_target = llm_pruner_target or runtime_summary.get("llm_pruner_target")
        llm_pruner_kept_channels_per_layer = (
            llm_pruner_kept_channels_per_layer
            or runtime_summary.get("llm_pruner_kept_channels_per_layer")
        )
        llm_pruner_pruned_channels_per_layer = (
            llm_pruner_pruned_channels_per_layer
            or runtime_summary.get("llm_pruner_pruned_channels_per_layer")
        )
        llm_pruner_actual_mlp_channel_pruning_ratio = (
            llm_pruner_actual_mlp_channel_pruning_ratio
            if llm_pruner_actual_mlp_channel_pruning_ratio is not None
            else runtime_summary.get("llm_pruner_actual_mlp_channel_pruning_ratio")
        )
        weight_sparsity_by_module = (
            weight_sparsity_by_module or runtime_summary.get("weight_sparsity_by_module")
        )
        matched_rasp_reference = matched_rasp_reference or runtime_summary.get("matched_rasp_reference")
        target_matched_to_rasp_actual_mlp_pruning = (
            target_matched_to_rasp_actual_mlp_pruning
            if target_matched_to_rasp_actual_mlp_pruning is not None
            else runtime_summary.get("target_matched_to_rasp_actual_mlp_pruning")
        )
        pruning_granularity = pruning_granularity or runtime_summary.get("pruning_granularity")
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
        checkpoint = runtime_summary.get("controller_checkpoint")
        if checkpoint:
            stage_risk_checkpoints.add(str(checkpoint))
        stage_risk_ratio_tokens.update(runtime_summary.get("selected_ratio_tokens", {}))
        stage_risk_mask_swaps += int(runtime_summary.get("mask_swap_count", 0))
        stage_risk_mask_swap_candidates += int(runtime_summary.get("mask_swap_candidates", 0))
        if "mean_mask_jaccard" in runtime_summary:
            stage_risk_mask_jaccards.append(float(runtime_summary["mean_mask_jaccard"]))
        for decision in runtime_summary.get("controller_decisions", []):
            stage = str(decision["stage"])
            stage_risk_actions[stage] += 1
            for ratio, risk in decision.get("risks", {}).items():
                stage_risk_risks[stage][str(ratio)].append(float(risk))
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
    protocol_valid = [row for row in rows if bool(row["stage_protocol"]["valid"])]
    fallback_rows = [row for row in rows if not bool(row["stage_protocol"]["valid"])]
    summary["protocol_valid_accuracy"] = (
        sum(int(row["correct"]) for row in protocol_valid) / len(protocol_valid)
        if protocol_valid else None
    )
    summary["fallback_accuracy"] = (
        sum(int(row["correct"]) for row in fallback_rows) / len(fallback_rows)
        if fallback_rows else None
    )
    summary["protocol_valid_problems"] = len(protocol_valid)
    summary["fallback_problems"] = len(fallback_rows)
    if stage_risk_checkpoints:
        if len(stage_risk_checkpoints) != 1:
            raise RuntimeError("A single method evaluation cannot mix stage-risk controller checkpoints")
        checkpoint = next(iter(stage_risk_checkpoints))
        summary["stage_risk_adaptive"] = {
            "actions_by_stage": dict(stage_risk_actions),
            "ratio_token_distribution": dict(stage_risk_ratio_tokens),
            "mean_risk_by_stage_ratio": {
                stage: {ratio: sum(values) / len(values) for ratio, values in ratios.items() if values}
                for stage, ratios in stage_risk_risks.items()
            },
            "dense_rate_by_stage": {
                stage: (
                    float(dense_observation_tokens[stage])
                    / float(dense_observation_tokens[stage] + masked_tokens[stage])
                    if dense_observation_tokens[stage] + masked_tokens[stage]
                    else None
                )
                for stage in STAGES
            },
            "mask_swap_count": stage_risk_mask_swaps,
            "mask_swap_candidates": stage_risk_mask_swap_candidates,
            "mean_mask_jaccard": (
                sum(stage_risk_mask_jaccards) / len(stage_risk_mask_jaccards)
                if stage_risk_mask_jaccards else None
            ),
            "controller_checkpoint": checkpoint,
            "controller_checkpoint_hash": file_sha256(checkpoint),
            "method_config_hash": stable_hash(method),
        }
    if "target_pruning_ratio" in method:
        target = float(method["target_pruning_ratio"])
        actual_value = float(summary["actual_average_mlp_pruning_ratio"])
        summary["target_pruning_ratio"] = target
        min_target = float(method.get("min_target_pruning_ratio", target))
        max_target = float(method.get("max_target_pruning_ratio", target))
        if min_target > max_target:
            raise ValueError(f"Invalid pruning target range: {min_target} > {max_target}")
        summary["min_target_pruning_ratio"] = min_target
        summary["max_target_pruning_ratio"] = max_target
        summary["target_pruning_reached"] = bool(actual_value >= min_target)
        summary["target_pruning_not_exceeded"] = bool(actual_value <= max_target)
        summary["target_pruning_range_passed"] = bool(min_target <= actual_value <= max_target)
        summary["target_pruning_gap"] = actual_value - target
        summary["target_pruning_status"] = (
            "passed"
            if min_target <= actual_value <= max_target
            else (
                "target_pruning_exceeded"
                if actual_value > max_target
                else "target_pruning_not_reached"
            )
        )
    if "target_pruning_label" in method:
        summary["target_pruning_label"] = method["target_pruning_label"]
    if runtime_backend:
        summary["runtime_backend"] = runtime_backend
    if runtime_baseline_type:
        summary["runtime_baseline_type"] = runtime_baseline_type
    if runtime_selection_method:
        summary["runtime_selection_method"] = runtime_selection_method
    if runtime_prune_ratio is not None:
        summary["runtime_prune_ratio"] = runtime_prune_ratio
    if runtime_density is not None:
        summary["runtime_density"] = runtime_density
    if prompt_dense_tokens:
        summary["prompt_dense_tokens"] = prompt_dense_tokens
    if decode_masked_tokens:
        summary["decode_masked_tokens"] = decode_masked_tokens
    if keep_ratios_by_layer:
        summary["keep_ratios_by_layer"] = keep_ratios_by_layer
    if pruning_granularity:
        summary["pruning_granularity"] = pruning_granularity
    if wanda_weight_sparsity is not None:
        summary["wanda_weight_sparsity"] = wanda_weight_sparsity
    if wanda_sparsity_ratio is not None:
        summary["wanda_sparsity_ratio"] = wanda_sparsity_ratio
    if wanda_calibration_samples is not None:
        summary["wanda_calibration_samples"] = wanda_calibration_samples
    if wanda_calibration_source:
        summary["wanda_calibration_source"] = wanda_calibration_source
    if wanda_calibration_prompt_mode:
        summary["wanda_calibration_prompt_mode"] = wanda_calibration_prompt_mode
    if wanda_calibration_text_field:
        summary["wanda_calibration_text_field"] = wanda_calibration_text_field
    if wanda_calibration_seed is not None:
        summary["wanda_calibration_seed"] = wanda_calibration_seed
    if wanda_target_modules:
        summary["wanda_target_modules"] = wanda_target_modules
    if sparsegpt_weight_sparsity is not None:
        summary["sparsegpt_weight_sparsity"] = sparsegpt_weight_sparsity
    if sparsegpt_sparsity_ratio is not None:
        summary["sparsegpt_sparsity_ratio"] = sparsegpt_sparsity_ratio
    if sparsegpt_calibration_samples is not None:
        summary["sparsegpt_calibration_samples"] = sparsegpt_calibration_samples
    if sparsegpt_calibration_source:
        summary["sparsegpt_calibration_source"] = sparsegpt_calibration_source
    if sparsegpt_target_modules:
        summary["sparsegpt_target_modules"] = sparsegpt_target_modules
    if sparsegpt_artifact_path:
        summary["sparsegpt_artifact_path"] = sparsegpt_artifact_path
    if sparsegpt_artifact_hash:
        summary["sparsegpt_artifact_hash"] = sparsegpt_artifact_hash
    if sparsegpt_blocksize is not None:
        summary["sparsegpt_blocksize"] = sparsegpt_blocksize
    if sparsegpt_percdamp is not None:
        summary["sparsegpt_percdamp"] = sparsegpt_percdamp
    if shortgpt_calibration_samples is not None:
        summary["shortgpt_calibration_samples"] = shortgpt_calibration_samples
    if shortgpt_calibration_source:
        summary["shortgpt_calibration_source"] = shortgpt_calibration_source
    if shortgpt_candidate_layers:
        summary["shortgpt_candidate_layers"] = shortgpt_candidate_layers
    if shortgpt_pruned_layers:
        summary["shortgpt_pruned_layers"] = shortgpt_pruned_layers
    if shortgpt_block_influence_by_layer:
        summary["shortgpt_block_influence_by_layer"] = shortgpt_block_influence_by_layer
    if shortgpt_layer_pruning_ratio is not None:
        summary["shortgpt_layer_pruning_ratio"] = shortgpt_layer_pruning_ratio
    if flap_metric:
        summary["flap_metric"] = flap_metric
    if flap_structure:
        summary["flap_structure"] = flap_structure
    if flap_prune_ratio is not None:
        summary["flap_prune_ratio"] = flap_prune_ratio
    if flap_calibration_dataset:
        summary["flap_calibration_dataset"] = flap_calibration_dataset
    if flap_calibration_samples is not None:
        summary["flap_calibration_samples"] = flap_calibration_samples
    if flap_calibration_source:
        summary["flap_calibration_source"] = flap_calibration_source
    if flap_bias_compensation is not None:
        summary["flap_bias_compensation"] = flap_bias_compensation
    if flap_physical_pruning is not None:
        summary["flap_physical_pruning"] = flap_physical_pruning
    if flap_target:
        summary["flap_target"] = flap_target
    if flap_kept_channels_per_layer:
        summary["flap_kept_channels_per_layer"] = flap_kept_channels_per_layer
    if flap_pruned_channels_per_layer:
        summary["flap_pruned_channels_per_layer"] = flap_pruned_channels_per_layer
    if flap_actual_mlp_channel_pruning_ratio is not None:
        summary["flap_actual_mlp_channel_pruning_ratio"] = flap_actual_mlp_channel_pruning_ratio
    if llm_pruner_importance:
        summary["llm_pruner_importance"] = llm_pruner_importance
    if llm_pruner_structure:
        summary["llm_pruner_structure"] = llm_pruner_structure
    if llm_pruner_prune_ratio is not None:
        summary["llm_pruner_prune_ratio"] = llm_pruner_prune_ratio
    if llm_pruner_physical_pruning is not None:
        summary["llm_pruner_physical_pruning"] = llm_pruner_physical_pruning
    if llm_pruner_target:
        summary["llm_pruner_target"] = llm_pruner_target
    if llm_pruner_kept_channels_per_layer:
        summary["llm_pruner_kept_channels_per_layer"] = llm_pruner_kept_channels_per_layer
    if llm_pruner_pruned_channels_per_layer:
        summary["llm_pruner_pruned_channels_per_layer"] = llm_pruner_pruned_channels_per_layer
    if llm_pruner_actual_mlp_channel_pruning_ratio is not None:
        summary["llm_pruner_actual_mlp_channel_pruning_ratio"] = (
            llm_pruner_actual_mlp_channel_pruning_ratio
        )
    if weight_sparsity_by_module:
        summary["weight_sparsity_by_module"] = weight_sparsity_by_module
    if matched_rasp_reference:
        summary["matched_rasp_reference"] = matched_rasp_reference
    if target_matched_to_rasp_actual_mlp_pruning is not None:
        summary["target_matched_to_rasp_actual_mlp_pruning"] = (
            target_matched_to_rasp_actual_mlp_pruning
        )
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
