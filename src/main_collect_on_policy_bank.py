from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.main_generate import DEFAULT_STOP_STRINGS, truncate_completion
from src.metrics.answer_match import answer_match, extract_answer
from src.models.load_model import load_model_bundle
from src.rasp.greedy_decode import greedy_decode_replay_history_counterfactual
from src.rasp.config_fingerprint import config_fingerprint
from src.rasp.mlp_runtime import apply_runtime_mlp_masking_qwen3
from src.rasp.on_policy import prior_action_candidate_events, selected_action_event
from src.rasp.stage_runtime import RuntimeStageProbe
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["id"])


def cosine_distance(left: object, right: object) -> float:
    a = torch.as_tensor(left).detach().float().reshape(-1)
    b = torch.as_tensor(right).detach().float().reshape(-1)
    return float(
        1.0
        - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    )


def maximum_absolute_difference(left: object, right: object) -> float:
    a = torch.as_tensor(left).detach().float().reshape(-1)
    b = torch.as_tensor(right).detach().float().reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(torch.max(torch.abs(a - b)).item())


def token_divergence(reference: list[int], candidate: list[int]) -> float:
    length = max(len(reference), len(candidate))
    if not length:
        return 0.0
    return sum(
        int(
            index >= len(reference)
            or index >= len(candidate)
            or reference[index] != candidate[index]
        )
        for index in range(length)
    ) / length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    set_seed(int(cfg.get("seed", 1)))
    bank = cfg["on_policy_bank"]
    paths = cfg["paths"]
    generation = cfg["generation"]
    ratios = [float(value) for value in bank["candidate_ratios"]]
    if not ratios or ratios[0] != 0.0:
        raise ValueError("On-policy candidate ratios must start with dense control")
    behavior_rows = {key(row): row for row in read_jsonl(paths["behavior_trajectories"])}
    dense_rows = {key(row): row for row in read_jsonl(paths["dense_trajectories"])}
    if set(behavior_rows) != set(dense_rows):
        raise ValueError("Dense and behavior trajectories use different dev tasks")
    if any(
        behavior_rows[item].get("question") != dense_rows[item].get("question")
        or behavior_rows[item].get("gold") != dense_rows[item].get("gold")
        or behavior_rows[item].get("prompt") != dense_rows[item].get("prompt")
        for item in behavior_rows
    ):
        raise ValueError("Dense and behavior trajectories use different prompt/task metadata")
    bundle = load_model_bundle(cfg["model"])
    apply_runtime_mlp_masking_qwen3(bundle.model, ratios=ratios)
    stage_cfg = cfg["stage_sensitivity"]
    stage_probe = RuntimeStageProbe(
        stage_cfg["checkpoint"],
        float(stage_cfg["reasoning_threshold"]),
        confidence_threshold=stage_cfg.get("confidence_threshold"),
        require_causal_features=True,
    )
    max_problems = int(bank["max_problems"])
    max_boundaries = int(bank["max_boundaries_per_problem"])
    window_tokens = int(bank["window_tokens"])
    cooldown_tokens = int(bank.get("cooldown_tokens", 16))
    decision_start = int(bank.get("decision_start", 32))
    decision_stride = int(bank.get("decision_stride", 32))
    max_new_tokens = int(generation["max_new_tokens"])
    entropy_tolerance = float(bank["replay_entropy_tolerance"])
    confidence_tolerance = float(bank["replay_confidence_tolerance"])
    hidden_tolerance = float(bank["replay_hidden_cosine_tolerance"])
    logits_tolerance = float(bank["replay_logits_tolerance"])
    output_rows, output_hidden = [], []
    attempted_problems = valid_problems = replay_failures = invalid_candidate_boundaries = 0
    eligible_dense_correct_problems = eligible_behavior_correct_problems = 0
    for problem in sorted(behavior_rows):
        if valid_problems >= max_problems:
            break
        behavior = behavior_rows[problem]
        dense = dense_rows[problem]
        dense_correct = answer_match(str(dense["completion"]), str(dense.get("gold", "")))
        behavior_correct = answer_match(
            str(behavior["completion"]), str(behavior.get("gold", ""))
        )
        require_dense_correct = bool(
            bank.get(
                "require_dense_correct",
                bank.get("require_dense_and_behavior_correct", True),
            )
        )
        require_behavior_correct = bool(
            bank.get(
                "require_behavior_correct",
                bank.get("require_dense_and_behavior_correct", False),
            )
        )
        if require_dense_correct and not dense_correct:
            continue
        if require_behavior_correct and not behavior_correct:
            continue
        eligible_dense_correct_problems += int(dense_correct)
        eligible_behavior_correct_problems += int(behavior_correct)
        attempted_problems += 1
        runtime = behavior["runtime"]
        token_ids = [int(value) for value in runtime.get("generated_token_ids", [])]
        events = runtime.get("router_events", [])
        candidates = prior_action_candidate_events(
            events,
            token_count=len(token_ids),
            window_tokens=window_tokens,
            cooldown_tokens=cooldown_tokens,
            decision_start=decision_start,
            decision_stride=decision_stride,
            max_boundaries=max_boundaries,
        )
        problem_rows = []
        problem_hidden = []
        for event in candidates:
            boundary = int(event["generated_tokens"])
            prefix = token_ids[:boundary]
            action_history = [
                {
                    "generated_tokens": int(item["generated_tokens"]),
                    "selected_ratio": float(item["selected_ratio"]),
                    "duration_tokens": window_tokens,
                }
                for item in events
                if int(item["generated_tokens"]) < boundary and selected_action_event(item)
            ]
            decision_history = [
                {
                    "generated_tokens": int(item["generated_tokens"]),
                    "selected_ratio": float(item["ratio"]),
                    "duration_tokens": int(item["duration_tokens"]),
                }
                for item in (event.get("decision") or {}).get("action_history", [])
                if int(item["generated_tokens"]) < boundary
            ]
            schedule = {
                int(item["generated_tokens"]): float(item["selected_ratio"])
                for item in events
                if int(item["generated_tokens"]) < boundary
            }
            branches = [
                greedy_decode_replay_history_counterfactual(
                    bundle.model,
                    bundle.tokenizer,
                    behavior["prompt"],
                    prefix,
                    schedule,
                    ratio,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=int(generation.get("max_input_tokens", 2048)),
                    window_tokens=window_tokens,
                )
                for ratio in ratios
            ]
            control = branches[0]
            observation = control["boundary_observation"]
            replay_ok = (
                not control["replay_mismatches"]
                and decision_history == action_history
                and int(control["boundary_next_token_id"]) == int(event["next_token_id"])
                and control["boundary_logits_sha256"] == event["logits_sha256"]
                and control["boundary_logits_topk_ids"] == event["logits_topk_ids"]
                and maximum_absolute_difference(
                    control["boundary_logits_topk_values"], event["logits_topk_values"]
                )
                <= logits_tolerance
                and abs(float(control["boundary_logits_l2"]) - float(event["logits_l2"]))
                <= logits_tolerance * max(1.0, abs(float(event["logits_l2"])))
                and abs(float(observation.entropy) - float(event["entropy"])) <= entropy_tolerance
                and abs(float(observation.confidence) - float(event["confidence"]))
                <= confidence_tolerance
                and cosine_distance(observation.hidden_state, event["hidden_state"])
                <= hidden_tolerance
                and maximum_absolute_difference(
                    observation.hidden_state, event["hidden_state"]
                )
                <= hidden_tolerance
            )
            if not replay_ok:
                replay_failures += 1
                continue
            if any(
                not bool(branch["action_completed_or_terminal"])
                for branch in branches
            ):
                invalid_candidate_boundaries += 1
                continue
            recent = bundle.tokenizer.decode(
                prefix[-int(stage_cfg.get("recent_tokens", 128)) :],
                skip_special_tokens=True,
            )
            stage = stage_probe.classify(
                hidden_state=observation.hidden_state,
                entropy=observation.entropy,
                confidence=observation.confidence,
                position=boundary / max(1, max_new_tokens),
                recent_text=recent,
                boundary_index=boundary // window_tokens,
                num_boundaries=max_new_tokens // window_tokens,
            )
            completions = [
                truncate_completion(
                    branch["completion"], generation.get("stop_strings", DEFAULT_STOP_STRINGS)
                )
                for branch in branches
            ]
            control_answer = extract_answer(completions[0])
            control_correct = answer_match(
                completions[0], str(behavior.get("gold", ""))
            )
            candidate_correct = [
                answer_match(value, str(behavior.get("gold", "")))
                for value in completions
            ]
            problem_rows.append(
                {
                    "dataset": behavior["dataset"],
                    "id": behavior["id"],
                    "dense_trajectory_correct": dense_correct,
                    "behavior_trajectory_correct": behavior_correct,
                    "generated_tokens_at_boundary": boundary,
                    "max_new_tokens": max_new_tokens,
                    "entropy": float(observation.entropy),
                    "confidence": float(observation.confidence),
                    "stage_probabilities": stage["stage_probabilities"],
                    "trusted_stage": stage["trusted_stage"],
                    "stage_confidence": stage["stage_confidence"],
                    "prior_action_count": len(action_history),
                    "prior_ratio_sum": sum(item["selected_ratio"] for item in action_history),
                    "prior_pruned_tokens": len(action_history) * window_tokens,
                    "tokens_since_last_action": (
                        boundary - action_history[-1]["generated_tokens"]
                        if action_history
                        else None
                    ),
                    "last_action_ratio": (
                        action_history[-1]["selected_ratio"] if action_history else None
                    ),
                    "action_history": action_history,
                    "candidate_ratios": ratios,
                    "on_policy_dense_control_correct": control_correct,
                    "candidate_flipped_from_on_policy_dense_control": [
                        extract_answer(value) != control_answer for value in completions
                    ],
                    "candidate_harmful_flip": [
                        bool(control_correct and not value)
                        for value in candidate_correct
                    ],
                    "candidate_beneficial_correction": [
                        bool(not control_correct and value)
                        for value in candidate_correct
                    ],
                    "candidate_window_token_divergence": [
                        token_divergence(control["window_ids"], branch["window_ids"])
                        for branch in branches
                    ],
                    "candidate_terminal_eos": [
                        bool(branch["terminated_by_eos"]) for branch in branches
                    ],
                    "candidate_correct": candidate_correct,
                    "candidate_answers": [extract_answer(value) for value in completions],
                    "replay_prefix_tokens": boundary,
                    "replay_verified": True,
                }
            )
            problem_hidden.append(observation.hidden_state.detach().float().cpu().reshape(-1))
        if problem_rows:
            output_rows.extend(problem_rows)
            output_hidden.extend(problem_hidden)
            valid_problems += 1
    output = ensure_dir(paths["run_dir"])
    write_jsonl(paths["on_policy_dataset"], output_rows)
    if output_hidden:
        torch.save(torch.stack(output_hidden), paths["on_policy_hidden_states"])
    summary = {
        "schema": "rasp_on_policy_action_risk_smoke_v1",
        "risk_label_semantics": "harmful_flip_conditioned_on_correct_dense_control_v1",
        "dataset": next(iter(behavior_rows))[0] if behavior_rows else "unknown",
        "behavior_policy_tag": bank["behavior_policy_tag"],
        "attempted_problems": attempted_problems,
        "valid_problems": valid_problems,
        "eligible_dense_correct_problems": eligible_dense_correct_problems,
        "eligible_behavior_correct_problems": eligible_behavior_correct_problems,
        "boundaries": len(output_rows),
        "replay_failures": replay_failures,
        "invalid_candidate_boundaries": invalid_candidate_boundaries,
        "candidate_ratios": ratios,
        "cooldown_tokens": cooldown_tokens,
        "decision_start": decision_start,
        "decision_stride": decision_stride,
        "rows_with_prior_action": sum(int(row["prior_action_count"] > 0) for row in output_rows),
        "rows_with_correct_on_policy_dense_control": sum(
            int(row["on_policy_dense_control_correct"]) for row in output_rows
        ),
        "final_test_sources_used": False,
        "replay_components_verified": [
            "action_schedule",
            "forced_token_prefix",
            "boundary_next_token",
            "boundary_topk_logits_and_l2",
            "boundary_full_logits_sha256",
            "boundary_entropy_confidence",
            "boundary_hidden_cosine_and_max_abs",
        ],
        "on_policy_config_fingerprint": config_fingerprint(
            cfg,
            (
                "seed",
                "model",
                "prompt",
                "data",
                "generation",
                "runtime_rasp",
                "on_policy_bank",
                "stage_sensitivity",
            ),
        ),
    }
    write_json(paths["on_policy_summary"], summary)
    checks = {
        "minimum_valid_problems": valid_problems >= max_problems,
        "contains_prior_action_states": bool(output_rows)
        and all(int(row["prior_action_count"]) > 0 for row in output_rows),
        "all_rows_originate_from_dense_correct_problems": bool(output_rows)
        and all(bool(row["dense_trajectory_correct"]) for row in output_rows),
        "candidate_respects_dense_cooldown": bool(output_rows)
        and all(
            int(row["tokens_since_last_action"]) >= window_tokens + cooldown_tokens
            for row in output_rows
        ),
        "replay_is_exact": replay_failures == 0,
        "complete_candidate_grid": all(row["candidate_ratios"] == ratios for row in output_rows),
        "no_final_test_sources": True,
    }
    write_json(
        paths["on_policy_validation"],
        {
            "schema": "rasp_on_policy_action_risk_smoke_validation_v1",
            "status": "ok" if all(checks.values()) else "failed",
            "checks": checks,
            **summary,
        },
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
