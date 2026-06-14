from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


def optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def gate_status(gate: dict[str, Any] | None, key: str = "passed") -> str:
    if gate is None:
        return "pending"
    return "passed" if bool(gate.get(key)) else "failed"


def artifact_gate_status(
    gate: dict[str, Any] | None, artifacts: tuple[Path, ...]
) -> str:
    status = gate_status(gate)
    if status == "passed" and not all(path.exists() for path in artifacts):
        return "failed"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--failed-stage")
    parser.add_argument("--active-stage")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    preflight = optional_json(root / "00_preflight" / "00_existing_bank_preflight.json")
    preflight_gate = optional_json(root / "00_preflight" / "phase_gate.json")
    smoke_gate = optional_json(root / "01_dense_bank_smoke" / "phase_gate.json")
    pilot_gate = optional_json(root / "02_dense_bank_pilot" / "phase_gate.json")
    pilot_data = optional_json(
        root / "02_dense_bank_pilot" / "data" / "01_full_trajectory_data_summary.json"
    )
    pilot_analysis = optional_json(
        root / "02_dense_bank_pilot" / "analysis" / "02_full_trajectory_analysis.json"
    )
    fixed = optional_json(root / "03_fixed_multi_window_dev" / "fixed_multi_window_summary.json")
    fixed_gate = optional_json(root / "03_fixed_multi_window_dev" / "phase_gate.json")
    behavior = optional_json(root / "03_fixed_multi_window_dev" / "selected_behavior_policy.json")
    on_policy = optional_json(root / "04_on_policy_smoke" / "on_policy_smoke_summary.json")
    on_policy_gate = optional_json(root / "04_on_policy_smoke" / "phase_gate.json")

    phase_status = {
        "existing_bank_preflight": (
            artifact_gate_status(
                preflight_gate,
                (root / "00_preflight" / "00_existing_bank_preflight.json",),
            )
            if preflight_gate is not None
            else (
                "pending"
                if preflight is None
                else ("passed" if preflight.get("preflight_valid") else "failed")
            )
        ),
        "dense_bank_smoke": artifact_gate_status(
            smoke_gate,
            (
                root
                / "01_dense_bank_smoke"
                / "data"
                / "01_full_trajectory_data_summary.json",
            ),
        ),
        "full_trajectory_pilot": artifact_gate_status(
            pilot_gate,
            (
                root
                / "02_dense_bank_pilot"
                / "data"
                / "01_full_trajectory_data_summary.json",
                root
                / "02_dense_bank_pilot"
                / "analysis"
                / "02_full_trajectory_analysis.json",
            ),
        ),
        "fixed_multi_window_dev": (
            artifact_gate_status(
                fixed_gate,
                (
                    root / "03_fixed_multi_window_dev" / "fixed_multi_window_summary.json",
                    root / "03_fixed_multi_window_dev" / "selected_behavior_policy.json",
                ),
            )
            if fixed_gate is not None
            else (
                "pending"
                if fixed is None
                else ("passed" if fixed.get("multi_window_feasible") else "failed")
            )
        ),
        "on_policy_smoke": artifact_gate_status(
            on_policy_gate,
            (root / "04_on_policy_smoke" / "on_policy_smoke_summary.json",),
        ),
    }
    completed = bool(
        all(value == "passed" for value in phase_status.values())
        and not args.failed_stage
        and not args.active_stage
    )
    failed_stage = args.failed_stage or next(
        (name for name, value in phase_status.items() if value == "failed"), None
    )
    workflow_gate = {
        "schema": "rasp_full_trajectory_multi_window_workflow_gate_v1",
        "completed": completed,
        "failed_stage": failed_stage,
        "active_stage": args.active_stage,
        "phase_status": phase_status,
        "on_policy_bank_expansion_allowed": bool(
            completed and on_policy and on_policy.get("on_policy_bank_expansion_allowed")
        ),
        "learned_multi_window_allowed": False,
    }
    failed_checks = {}
    for name, gate in (
        ("existing_bank_preflight", preflight_gate),
        ("dense_bank_smoke", smoke_gate),
        ("full_trajectory_pilot", pilot_gate),
        ("fixed_multi_window_dev", fixed_gate),
        ("on_policy_smoke", on_policy_gate),
    ):
        if gate and not gate.get("passed"):
            failed_checks[name] = [
                check for check, value in gate.get("checks", {}).items() if not value
            ]
    if fixed and not fixed.get("multi_window_feasible") and not fixed_gate:
        failed_checks["fixed_multi_window_dev"] = [
            "No policy passed the two-source risk/exposure gate."
        ]
    required_artifacts = {
        "existing_bank_preflight": (
            root / "00_preflight" / "00_existing_bank_preflight.json",
        ),
        "dense_bank_smoke": (
            root
            / "01_dense_bank_smoke"
            / "data"
            / "01_full_trajectory_data_summary.json",
        ),
        "full_trajectory_pilot": (
            root
            / "02_dense_bank_pilot"
            / "data"
            / "01_full_trajectory_data_summary.json",
            root
            / "02_dense_bank_pilot"
            / "analysis"
            / "02_full_trajectory_analysis.json",
        ),
        "fixed_multi_window_dev": (
            root / "03_fixed_multi_window_dev" / "fixed_multi_window_summary.json",
            root / "03_fixed_multi_window_dev" / "selected_behavior_policy.json",
        ),
        "on_policy_smoke": (
            root / "04_on_policy_smoke" / "on_policy_smoke_summary.json",
        ),
    }
    for name, paths in required_artifacts.items():
        missing = [str(path) for path in paths if not path.exists()]
        if phase_status[name] == "failed" and missing:
            failed_checks.setdefault(name, []).extend(
                f"missing artifact: {path}" for path in missing
            )
    workflow_gate["failed_checks"] = failed_checks
    write_json(root / "workflow_gate.json", workflow_gate)

    final_summary = {
        "schema": "rasp_full_trajectory_multi_window_final_summary_v1",
        **workflow_gate,
        "full_trajectory_coverage": (
            {
                "source_eligible_problems": pilot_data.get("source_eligible_problems"),
                "causal_boundary_count": pilot_data.get("causal_boundary_count"),
                "tail_diagnostic_boundary_count": pilot_data.get(
                    "tail_diagnostic_boundary_count"
                ),
                "causal_position_decile_counts": (
                    pilot_analysis.get("causal_position_decile_counts")
                    if pilot_analysis
                    else None
                ),
                "diagnostic_dense_trajectory_position_decile_counts": (
                    pilot_analysis.get(
                        "diagnostic_dense_trajectory_position_decile_counts"
                    )
                    if pilot_analysis
                    else None
                ),
            }
            if pilot_data
            else None
        ),
        "soft_stage_unknown_distribution": (
            {
                "trusted_stage_counts": pilot_data.get("trusted_stage_counts"),
                "mean_soft_stage_probabilities": pilot_data.get(
                    "mean_soft_stage_probabilities"
                ),
            }
            if pilot_data
            else None
        ),
        "dense_state_ratio_dose_response": pilot_data.get("dose_response") if pilot_data else None,
        "trusted_stage_ratio_response": (
            pilot_data.get("trusted_stage_ratio_response") if pilot_data else None
        ),
        "dense_state_oof_risk_prediction": (
            {
                "positive_flips": pilot_analysis.get("positive_flips"),
                "variants": pilot_analysis.get("variants"),
                "context_beats_action_both_metrics_fold_count": pilot_analysis.get(
                    "context_beats_action_both_metrics_fold_count"
                ),
                "soft_stage_beats_context_both_metrics_fold_count": pilot_analysis.get(
                    "soft_stage_beats_context_both_metrics_fold_count"
                ),
                "hidden_beats_context_both_metrics_fold_count": pilot_analysis.get(
                    "hidden_beats_context_both_metrics_fold_count"
                ),
                "soft_stage_adds_to_hidden_both_metrics_fold_count": pilot_analysis.get(
                    "soft_stage_adds_to_hidden_both_metrics_fold_count"
                ),
                "dense_risk_model_promising": pilot_analysis.get(
                    "dense_risk_model_promising"
                ),
            }
            if pilot_analysis
            else None
        ),
        "fixed_multi_window_risk_exposure_frontier": fixed.get("cells") if fixed else None,
        "selected_behavior_policy": (
            behavior.get("selected_behavior_policy") if behavior else None
        ),
        "on_policy_replay_integrity": on_policy,
        "logical_mask_only": True,
        "real_speedup_claimed": False,
        "final_test_sources_used": False,
        "learned_multi_window_allowed": False,
        "learned_multi_window_block_reason": (
            "This workflow stops at on-policy smoke. Expand the bank and pass grouped OOF first."
        ),
    }
    write_json(root / "final_workflow_summary.json", final_summary)
    oof_brief = (
        {
            name: {
                "roc_auc": value.get("roc_auc"),
                "pr_auc": value.get("pr_auc"),
            }
            for name, value in pilot_analysis.get("variants", {}).items()
        }
        if pilot_analysis
        else None
    )
    frontier_brief = (
        [
            {
                "dataset": cell["dataset"],
                "tag": cell["tag"],
                "exposure": cell["average_theoretical_pruning_exposure"],
                "accuracy_delta": cell["paired_accuracy_delta"],
                "dense_correct_flip_rate": cell["dense_correct_flip_rate"],
            }
            for cell in fixed.get("cells", [])
        ]
        if fixed
        else None
    )

    lines = [
        "# Full-Trajectory Multi-Window 工作流报告",
        "",
        f"- 工作流完成：`{completed}`",
        f"- 当前阶段：`{args.active_stage or 'none'}`",
        f"- 失败阶段：`{failed_stage or 'none'}`",
        f"- 失败检查：`{failed_checks or 'none'}`",
        f"- 可扩大 on-policy bank：`{workflow_gate['on_policy_bank_expansion_allowed']}`",
        "- 允许训练 learned multi-window：`false`",
        "",
        "## 阶段 Gate",
        "",
    ]
    lines.extend(f"- {name}: `{status}`" for name, status in phase_status.items())
    lines.extend(
        [
            "",
            "## 关键结果",
            "",
            f"- 完整轨迹覆盖：`{final_summary['full_trajectory_coverage']}`",
            f"- soft-stage / unknown 分布：`{final_summary['soft_stage_unknown_distribution']}`",
            f"- dense-state ratio 剂量响应：`{final_summary['dense_state_ratio_dose_response']}`",
            f"- trusted-stage × ratio 风险：`{final_summary['trusted_stage_ratio_response']}`",
            f"- grouped OOF 风险预测：`{oof_brief}`",
            f"- 固定多窗口风险—曝光前沿：`{frontier_brief}`",
            f"- 选中的 behavior policy：`{final_summary['selected_behavior_policy']}`",
            f"- on-policy replay 完整性：`{on_policy.get('checks') if on_policy else None}`",
            f"- on-policy ratio 剂量响应：`{on_policy.get('candidate_dose_response') if on_policy else None}`",
            "",
            "## 结论",
            "",
            "本轮只验证完整轨迹 dense-state bank、固定多窗口行为策略和 on-policy 精确重放。",
            "即使全部 Gate 通过，也必须扩大 on-policy bank 并通过 problem-grouped OOF，才允许训练 learned multi-window controller。",
        ]
    )
    (root / "final_workflow_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
