from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.metrics.answer_match import extract_answer
from src.models.load_model import load_model_bundle
from src.rasp.greedy_decode import greedy_decode_single_window_counterfactual
from src.rasp.mlp_runtime import apply_runtime_mlp_masking_qwen3
from src.rasp.stage_runtime import RuntimeStageProbe
from src.rasp.window_sampling import boundary_positions
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def token_divergence(reference: list[int], candidate: list[int]) -> float:
    length = max(len(reference), len(candidate))
    if length == 0:
        return 0.0
    return sum(
        int(index >= len(reference) or index >= len(candidate) or reference[index] != candidate[index])
        for index in range(length)
    ) / length


def hidden_drift(reference: torch.Tensor | None, candidate: torch.Tensor | None) -> dict[str, float | None]:
    if reference is None or candidate is None:
        return {"window_end_hidden_l2": None, "window_end_hidden_cosine_distance": None}
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    return {
        "window_end_hidden_l2": float(torch.linalg.vector_norm(cand - ref).item()),
        "window_end_hidden_cosine_distance": float(
            1.0 - torch.nn.functional.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    set_seed(int(cfg.get("seed", 1)))
    bank_cfg = cfg.get("aligned_window_bank", {})
    paths = cfg["paths"]
    ratios = [float(value) for value in bank_cfg.get("ratios", [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40])]
    if not ratios or abs(ratios[0]) > 1e-12:
        raise ValueError("Aligned window bank ratios must start with dense ratio=0 control")
    window_tokens = int(bank_cfg.get("window_tokens", 16))
    max_boundaries = bank_cfg.get("max_boundaries_per_example")
    boundary_sampling = str(bank_cfg.get("boundary_sampling", "prefix"))
    stage_cfg = cfg.get("stage_sensitivity")
    stage_probe = None
    recent_stage_tokens = 128
    if stage_cfg:
        stage_probe = RuntimeStageProbe(
            str(stage_cfg["checkpoint"]),
            float(stage_cfg["reasoning_threshold"]),
        )
        recent_stage_tokens = int(stage_cfg.get("recent_tokens", 128))
    generation_cfg = cfg.get("generation", {})
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 512))
    max_input_tokens = int(generation_cfg.get("max_input_tokens", 2048))

    bundle = load_model_bundle(cfg["model"])
    apply_runtime_mlp_masking_qwen3(bundle.model, ratios=ratios)
    rows, probe_rows, hidden_states = [], [], []
    trajectories = [row for row in read_jsonl(paths["trajectories"]) if bool(row.get("correct"))]
    for item in tqdm(trajectories, desc="aligned-window-bank"):
        prompt = build_prompt(item["question"], bundle.tokenizer, cfg.get("prompt", {}))
        baseline_ids = item.get("generated_token_ids")
        token_source = "trajectory_generated_token_ids"
        if baseline_ids is None:
            baseline_ids = bundle.tokenizer(item["completion"], add_special_tokens=False)["input_ids"]
            token_source = "retokenized_completion_fallback"
        positions = boundary_positions(
            len(baseline_ids),
            window_tokens,
            max_boundaries,
            boundary_sampling,
        )
        for boundary_index, position in enumerate(positions):
            if position >= max_new_tokens:
                continue
            forced_prefix = [int(value) for value in baseline_ids[:position]]
            dense = greedy_decode_single_window_counterfactual(
                bundle.model,
                bundle.tokenizer,
                prompt,
                forced_prefix,
                0.0,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                window_tokens=window_tokens,
            )
            dense_answer = extract_answer(dense["completion"])
            baseline_answer = extract_answer(item["completion"])
            stage_annotation = {}
            if stage_probe is not None:
                dense_observation = dense["boundary_observation"]
                stage_position = position / max(1, len(baseline_ids) - 1)
                recent_text = bundle.tokenizer.decode(
                    forced_prefix[-recent_stage_tokens:],
                    skip_special_tokens=True,
                )
                stage_annotation = stage_probe.classify(
                    hidden_state=dense_observation.hidden_state,
                    entropy=dense_observation.entropy,
                    confidence=dense_observation.confidence,
                    position=stage_position,
                    recent_text=recent_text,
                    boundary_index=boundary_index,
                    num_boundaries=len(positions),
                )
                stage_annotation["stage_position"] = stage_position
                stage_annotation["stage_position_definition"] = (
                    "generated_tokens_over_dense_trajectory_tokens_minus_one"
                )
            for ratio in ratios:
                result = dense if abs(ratio) < 1e-12 else greedy_decode_single_window_counterfactual(
                    bundle.model,
                    bundle.tokenizer,
                    prompt,
                    forced_prefix,
                    ratio,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    window_tokens=window_tokens,
                )
                answer = extract_answer(result["completion"])
                observation = result["boundary_observation"]
                row: dict[str, Any] = {
                    "id": item["id"],
                    "dataset": item.get("dataset"),
                    "boundary_index": boundary_index,
                    "segment_id": boundary_index,
                    "generated_tokens_at_boundary": position,
                    "position": position / max(1, max_new_tokens),
                    "max_new_tokens": max_new_tokens,
                    "window_tokens": window_tokens,
                    "action_duration_tokens": min(window_tokens, len(result["window_ids"])),
                    "action_scope": "single_fixed_window_then_dense",
                    "action_window_alignment": "affected_next_token_decisions_v2",
                    "ranking_scope": "initial_prompt_prefill_fixed",
                    "boundary_token_source": token_source,
                    "module": "mlp_intermediate_channels",
                    "unit": "mlp_intermediate_channels",
                    "ratio": ratio,
                    "baseline_answer": baseline_answer,
                    "dense_control_answer": dense_answer,
                    "dense_control_flipped_from_baseline": dense_answer != baseline_answer,
                    "counterfactual_answer": answer,
                    "flipped": answer != dense_answer,
                    "window_token_divergence": token_divergence(dense["window_ids"], result["window_ids"]),
                    "entropy": observation.entropy,
                    "confidence": observation.confidence,
                    **stage_annotation,
                    **hidden_drift(dense["window_end_hidden"], result["window_end_hidden"]),
                }
                rows.append(row)
                probe = dict(row)
                probe["hidden_index"] = len(hidden_states)
                hidden_states.append(observation.hidden_state.squeeze(0))
                probe_rows.append(probe)
    write_jsonl(paths["counterfactuals"], rows)
    write_jsonl(paths["probe_dataset"], probe_rows)
    ensure_dir(Path(paths["probe_hidden_states"]).parent)
    if not hidden_states:
        raise ValueError("Aligned window bank produced no boundary states")
    torch.save(torch.stack(hidden_states), paths["probe_hidden_states"])
    write_json(
        paths["aligned_window_bank_summary"],
        {
            "method": "rasp_phase_b_aligned_window_bank_v2",
            "dense_correct_trajectories": len(trajectories),
            "boundaries": len(rows) // len(ratios),
            "counterfactual_rows": len(rows),
            "ratios": ratios,
            "window_tokens": window_tokens,
            "max_new_tokens": max_new_tokens,
            "configured_max_boundaries_per_example": max_boundaries,
            "boundary_sampling": boundary_sampling,
            "action_scope": "single_fixed_window_then_dense",
            "action_window_alignment": "affected_next_token_decisions_v2",
            "ranking_scope": "initial_prompt_prefill_fixed",
            "boundary_token_sources": sorted({row["boundary_token_source"] for row in rows}),
            "stage_sensitivity_enabled": stage_probe is not None,
            "stage_probe_checkpoint": str(stage_cfg["checkpoint"]) if stage_cfg else None,
            "stage_reasoning_threshold": (
                float(stage_cfg["reasoning_threshold"]) if stage_cfg else None
            ),
            "stage_sensitivity_diagnostic_only": (
                bool(stage_cfg.get("diagnostic_only", False)) if stage_cfg else None
            ),
            "stage_position_definition": (
                "generated_tokens_over_dense_trajectory_tokens_minus_one"
                if stage_cfg
                else None
            ),
            "s1_5_controller_gate_passed": (
                bool(stage_cfg.get("s1_5_controller_gate_passed", False)) if stage_cfg else None
            ),
            "operational_stage_counts": (
                {
                    stage: sum(
                        1
                        for row in rows
                        if abs(float(row["ratio"])) < 1e-12
                        and row.get("operational_stage") == stage
                    )
                    for stage in ("setup", "reasoning", "verification", "final")
                }
                if stage_probe is not None
                else None
            ),
        },
    )


if __name__ == "__main__":
    main()
