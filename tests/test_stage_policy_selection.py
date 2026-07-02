from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stage_calibration.policy_selection import (
    aggregate_methods,
    build_policy_selection,
    load_downstream_methods_from_selection,
)
from src.stage_calibration.protocol import STAGES
from src.utils.io import write_json


STRUCTURED_PROMPT = {
    "use_chat_template": True,
    "enable_thinking": True,
    "explicit_stage_protocol": True,
    "forced_assistant_prefix": "[[STAGE_SETUP]]\n",
}


def ratios(value: float) -> dict[str, float]:
    return {stage: value for stage in STAGES}


def method_summary(
    name: str,
    policy: str,
    stage_ratios: dict[str, float],
    *,
    seed: int,
    accuracy: float,
    pruning: float,
    protocol: float = 0.95,
    fallback: float = 0.02,
    truncation: float = 0.0,
) -> dict:
    return {
        "method": {
            "name": name,
            "policy": policy,
            "stage_ratios": stage_ratios,
            "prompt": STRUCTURED_PROMPT,
            "bias_compensation": True,
        },
        "seed": seed,
        "problems": 100,
        "correct": int(round(accuracy * 100)),
        "accuracy": accuracy,
        "valid_stage_protocol_rate": protocol,
        "fallback_rate": fallback,
        "fallback_reasons": {},
        "truncation_rate": truncation,
        "mean_generated_tokens": 100,
        "stage_tokens": {},
        "theoretical_average_mlp_pruning_ratio": pruning,
    }


def seed_run(seed: int, structured_accuracy: float, candidate_accuracies: dict[str, float]) -> dict:
    methods = [
        method_summary(
            "ordinary_dense",
            "trajectory_global",
            ratios(0.0),
            seed=seed,
            accuracy=structured_accuracy + 0.01,
            pruning=0.0,
        ),
        method_summary(
            "structured_dense",
            "trajectory_global",
            ratios(0.0),
            seed=seed,
            accuracy=structured_accuracy,
            pruning=0.0,
        ),
        method_summary(
            "trajectory_global_0p10",
            "trajectory_global",
            ratios(0.10),
            seed=seed,
            accuracy=candidate_accuracies["trajectory_global_0p10"],
            pruning=0.10,
        ),
        method_summary(
            "stage_specific_0p20",
            "stage_specific",
            ratios(0.20),
            seed=seed,
            accuracy=candidate_accuracies["stage_specific_0p20"],
            pruning=0.20,
        ),
        method_summary(
            "shuffled_stage_0p10",
            "shuffled_stage",
            ratios(0.10),
            seed=seed,
            accuracy=candidate_accuracies["shuffled_stage_0p10"],
            pruning=0.10,
        ),
        method_summary(
            "stage_specific_0p30",
            "stage_specific",
            ratios(0.30),
            seed=seed,
            accuracy=candidate_accuracies["stage_specific_0p30"],
            pruning=0.30,
        ),
        method_summary(
            "coordinate_r0_setup_0p20",
            "stage_specific",
            {"setup": 0.2, "reasoning": 0.0, "verify": 0.0, "final": 0.0},
            seed=seed,
            accuracy=structured_accuracy,
            pruning=0.06,
        ),
    ]
    return {
        "root": f"run_seed{seed}",
        "seed": seed,
        "summary_path": f"run_seed{seed}/05_dev/summary.json",
        "frozen_policy_path": f"run_seed{seed}/05_dev/frozen_policy.json",
        "summary_sha256": f"summary-{seed}",
        "frozen_policy_sha256": f"frozen-{seed}",
        "summary": {
            "methods": methods,
            "prompt_gate": {"passed": True},
        },
        "frozen_policy": {},
        "ordinary_dense_accuracy": structured_accuracy + 0.01,
        "structured_dense_accuracy": structured_accuracy,
        "prompt_gate_passed": True,
    }


class StagePolicySelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runs = [
            seed_run(
                1,
                0.90,
                {
                    "trajectory_global_0p10": 0.87,
                    "stage_specific_0p20": 0.84,
                    "shuffled_stage_0p10": 0.89,
                    "stage_specific_0p30": 0.80,
                },
            ),
            seed_run(
                2,
                0.88,
                {
                    "trajectory_global_0p10": 0.85,
                    "stage_specific_0p20": 0.81,
                    "shuffled_stage_0p10": 0.87,
                    "stage_specific_0p30": 0.78,
                },
            ),
        ]

    def test_aggregates_cross_seed_drop_and_tier(self) -> None:
        aggregates = aggregate_methods(self.runs)
        by_name = {row["method_name"]: row for row in aggregates}
        self.assertEqual(by_name["trajectory_global_0p10"]["selection_tier"], "formal_safe")
        self.assertAlmostEqual(
            by_name["trajectory_global_0p10"]["mean_accuracy_drop_vs_structured_dense"],
            0.03,
        )
        self.assertEqual(by_name["stage_specific_0p20"]["selection_tier"], "main_candidate")
        self.assertEqual(by_name["stage_specific_0p30"]["selection_tier"], "aggressive_boundary")

    def test_selection_keeps_shuffled_as_control_not_main(self) -> None:
        selection = build_policy_selection(self.runs)
        selected = selection["selected_policies"]
        self.assertEqual(selected["conservative"]["method"]["original_method_name"], "trajectory_global_0p10")
        self.assertEqual(selected["main_dynamic"]["method"]["original_method_name"], "stage_specific_0p20")
        self.assertEqual(selected["shuffled_control"]["method"]["original_method_name"], "shuffled_stage_0p10")
        self.assertNotEqual(
            selected["conservative"]["method"]["original_method_name"],
            "coordinate_r0_setup_0p20",
        )
        self.assertNotEqual(
            selected["main_dynamic"]["method"]["policy"],
            "shuffled_stage",
        )

    def test_downstream_methods_have_policy_metadata_and_budget_presets(self) -> None:
        selection = build_policy_selection(self.runs)
        methods = {row["name"]: row for row in selection["downstream_methods"]}
        self.assertIn("ordinary_dense", methods)
        self.assertIn("structured_dense", methods)
        self.assertIn("stage_budget_conservative", methods)
        self.assertIn("main_dynamic_stage_specific_0p20", methods)
        self.assertEqual(methods["main_dynamic_stage_specific_0p20"]["selection_role"], "main_dynamic")
        self.assertEqual(methods["stage_budget_conservative"]["selection_source"], "predeclared_stage_budget_grid")

    def test_policy_selection_loader_rejects_test_consulted_artifact(self) -> None:
        selection = build_policy_selection(self.runs)
        selection["test_sets_consulted"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_selection.json"
            write_json(path, selection)
            with self.assertRaisesRegex(ValueError, "final test sets"):
                load_downstream_methods_from_selection(path)


if __name__ == "__main__":
    unittest.main()
