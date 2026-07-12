from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stage_calibration.policy_selection import (
    aggregate_methods,
    build_main_only_policy_selection,
    build_policy_selection,
    load_downstream_methods_from_selection,
)
from src.stage_calibration.protocol import STAGES
from src.utils.io import write_json

try:
    from src.main_stage_calibrated_pruning import _build_calibration_gate, adaptive_griffin_methods

    MAIN_WORKFLOW_AVAILABLE = True
except ModuleNotFoundError:
    _build_calibration_gate = None
    adaptive_griffin_methods = None
    MAIN_WORKFLOW_AVAILABLE = False


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
    if "stage_specific_0p10" in candidate_accuracies:
        methods.append(
            method_summary(
                "stage_specific_0p10",
                "stage_specific",
                ratios(0.10),
                seed=seed,
                accuracy=candidate_accuracies["stage_specific_0p10"],
                pruning=0.10,
            )
        )
    if "stage_specific_0p15" in candidate_accuracies:
        methods.append(
            method_summary(
                "stage_specific_0p15",
                "stage_specific",
                ratios(0.15),
                seed=seed,
                accuracy=candidate_accuracies["stage_specific_0p15"],
                pruning=0.15,
            )
        )
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

    def test_main_only_selection_chooses_highest_passing_dynamic_ratio(self) -> None:
        runs = [
            seed_run(
                3,
                0.90,
                {
                    "trajectory_global_0p10": 0.88,
                    "stage_specific_0p10": 0.88,
                    "stage_specific_0p15": 0.86,
                    "stage_specific_0p20": 0.83,
                    "shuffled_stage_0p10": 0.89,
                    "stage_specific_0p30": 0.80,
                },
            )
        ]
        selection = build_main_only_policy_selection(runs)
        methods = [row["name"] for row in selection["downstream_methods"]]
        self.assertEqual(methods, ["structured_dense", "dynamic_stage_main", "static_matched_global"])
        dynamic = selection["selected_policies"]["dynamic_stage_main"]["method"]
        static = selection["selected_policies"]["static_matched_global"]["method"]
        self.assertEqual(dynamic["original_method_name"], "stage_specific_0p15")
        self.assertEqual(dynamic["stage_ratios"], ratios(0.15))
        self.assertEqual(static["policy"], "trajectory_global")
        self.assertEqual(static["stage_ratios"], ratios(0.15))
        self.assertFalse(selection["test_sets_consulted"])

    def test_main_only_loader_rejects_extra_methods(self) -> None:
        selection = build_main_only_policy_selection(
            [
                seed_run(
                    3,
                    0.90,
                    {
                        "trajectory_global_0p10": 0.88,
                        "stage_specific_0p10": 0.88,
                        "stage_specific_0p15": 0.86,
                        "stage_specific_0p20": 0.83,
                        "shuffled_stage_0p10": 0.89,
                        "stage_specific_0p30": 0.80,
                    },
                )
            ]
        )
        selection["downstream_methods"].append(
            method_summary(
                "shuffled_stage_0p10",
                "shuffled_stage",
                ratios(0.10),
                seed=1,
                accuracy=0.9,
                pruning=0.1,
            )["method"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_selection_main_only.json"
            write_json(path, selection)
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_downstream_methods_from_selection(path)

    def test_adaptive_griffin_loader_accepts_only_main_methods(self) -> None:
        selection = {
            "schema": "stage_policy_selection_v1",
            "selection_mode": "adaptive_griffin_main_only",
            "test_sets_consulted": False,
            "downstream_methods": [
                {
                    "name": "structured_dense",
                    "policy": "trajectory_global",
                    "stage_ratios": ratios(0.0),
                    "prompt": STRUCTURED_PROMPT,
                },
                {
                    "name": "calibrated_stage_adaptive_griffin_main",
                    "policy": "calibrated_stage_adaptive_griffin",
                    "stage_ratios": {"setup": 0.2, "reasoning": 0.1, "verify": 0.2, "final": 0.0},
                    "prompt": STRUCTURED_PROMPT,
                    "alpha": 0.7,
                    "warmup_tokens": {"setup": 0, "reasoning": 16, "verify": 16, "final": 0},
                },
                {
                    "name": "static_matched_global",
                    "policy": "trajectory_global",
                    "stage_ratios": ratios(0.15),
                    "prompt": STRUCTURED_PROMPT,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adaptive_griffin_policy_selection.json"
            write_json(path, selection)
            methods, _ = load_downstream_methods_from_selection(path)
            self.assertEqual(
                [method["name"] for method in methods],
                ["structured_dense", "calibrated_stage_adaptive_griffin_main", "static_matched_global"],
            )
            selection["downstream_methods"].append(
                {
                    "name": "stage_specific_0p20",
                    "policy": "stage_specific",
                    "stage_ratios": ratios(0.20),
                    "prompt": STRUCTURED_PROMPT,
                }
            )
            write_json(path, selection)
            with self.assertRaisesRegex(ValueError, "Adaptive GRIFFIN"):
                load_downstream_methods_from_selection(path)

            selection["downstream_methods"] = selection["downstream_methods"][:3]
            selection["downstream_methods"][1] = {
                "name": "calibrated_stage_safe_dynamic_griffin_main",
                "policy": "calibrated_stage_safe_dynamic_griffin",
                "stage_ratios": {"setup": 0.15, "reasoning": 0.2, "verify": 0.15, "final": 0.0},
                "prompt": STRUCTURED_PROMPT,
                "runtime_weight": 0.4,
                "prior_weight": 0.6,
                "protected_core_ratios": {"setup": 0.5, "reasoning": 0.4, "verify": 0.6, "final": 1.0},
                "refresh_intervals": {"setup": 128, "reasoning": 64, "verify": 32, "final": 0},
            }
            write_json(path, selection)
            methods, _ = load_downstream_methods_from_selection(path)
            self.assertEqual(methods[1]["name"], "calibrated_stage_safe_dynamic_griffin_main")

            selection["downstream_methods"] = selection["downstream_methods"][:3]
            selection["downstream_methods"][1] = {
                "name": "static_core_residual_v4_3_final_light",
                "policy": "calibrated_stage_static_core_residual_griffin",
                "stage_ratios": {"setup": 0.2, "reasoning": 0.2, "verify": 0.1, "final": 0.1},
                "prompt": STRUCTURED_PROMPT,
                "runtime_weight": 0.3,
                "prior_weight": 0.7,
                "static_core_ratios": {"setup": 0.9, "reasoning": 0.92, "verify": 0.97, "final": 0.995},
                "swap_ratios": {"setup": 0.03, "reasoning": 0.025, "verify": 0.005, "final": 0.002},
            }
            write_json(path, selection)
            methods, _ = load_downstream_methods_from_selection(path)
            self.assertEqual(methods[1]["name"], "static_core_residual_v4_3_final_light")

    def test_adaptive_griffin_sweep_loader_accepts_multiple_adaptive_methods(self) -> None:
        selection = {
            "schema": "stage_policy_selection_v1",
            "selection_mode": "adaptive_griffin_sweep",
            "test_sets_consulted": False,
            "downstream_methods": [
                {
                    "name": "structured_dense",
                    "policy": "trajectory_global",
                    "stage_ratios": ratios(0.0),
                    "prompt": STRUCTURED_PROMPT,
                },
                {
                    "name": "safe_dynamic_v2_current",
                    "policy": "calibrated_stage_safe_dynamic_griffin",
                    "stage_ratios": {"setup": 0.15, "reasoning": 0.2, "verify": 0.15, "final": 0.0},
                    "prompt": STRUCTURED_PROMPT,
                },
                {
                    "name": "static_core_residual_stage_dynamic",
                    "policy": "calibrated_stage_static_core_residual_griffin",
                    "stage_ratios": {"setup": 0.2, "reasoning": 0.15, "verify": 0.1, "final": 0.0},
                    "prompt": STRUCTURED_PROMPT,
                },
                {
                    "name": "static_matched_global",
                    "policy": "trajectory_global",
                    "stage_ratios": ratios(0.15),
                    "prompt": STRUCTURED_PROMPT,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adaptive_griffin_sweep_selection.json"
            write_json(path, selection)
            methods, _ = load_downstream_methods_from_selection(path)
            self.assertEqual(
                [method["name"] for method in methods],
                [
                    "structured_dense",
                    "safe_dynamic_v2_current",
                    "static_core_residual_stage_dynamic",
                    "static_matched_global",
                ],
            )

            selection["downstream_methods"][2] = {
                "name": "stage_specific_0p20",
                "policy": "stage_specific",
                "stage_ratios": ratios(0.20),
                "prompt": STRUCTURED_PROMPT,
            }
            write_json(path, selection)
            with self.assertRaisesRegex(ValueError, "middle methods"):
                load_downstream_methods_from_selection(path)

            selection["downstream_methods"][2] = selection["downstream_methods"][1]
            write_json(path, selection)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_downstream_methods_from_selection(path)

    @unittest.skipUnless(MAIN_WORKFLOW_AVAILABLE, "main workflow dependencies are required")
    def test_main_only_workflow_can_disable_reference_calibration_gate(self) -> None:
        cfg = {
            "workflow": {"profile": "pilot"},
            "profiles": {"pilot": {}},
            "evaluation": {"require_calibration_reference_gate": False},
        }
        assert _build_calibration_gate is not None
        gate = _build_calibration_gate(cfg, [])
        self.assertFalse(gate["required"])
        self.assertTrue(gate["trajectory_calibration_promising"])
        self.assertIn("no calibration reference comparison", gate["skipped_reason"])

    @unittest.skipUnless(MAIN_WORKFLOW_AVAILABLE, "main workflow dependencies are required")
    def test_adaptive_griffin_methods_expand_variants(self) -> None:
        cfg = {
            "prompt": {"structured": STRUCTURED_PROMPT},
            "adaptive_griffin": {
                "enabled": True,
                "policy": "calibrated_stage_safe_dynamic_griffin",
                "stage_ratios": {"setup": 0.15, "reasoning": 0.2, "verify": 0.15, "final": 0.0},
                "protected_core_ratios": {"setup": 0.5, "reasoning": 0.4, "verify": 0.6, "final": 1.0},
                "refresh_intervals": {"setup": 128, "reasoning": 64, "verify": 32, "final": 0},
                "window_tokens": {"setup": 128, "reasoning": 64, "verify": 32, "final": 1},
                "runtime_weight": 0.4,
                "prior_weight": 0.6,
                "variants": [
                    {"method_name": "safe_dynamic_v2_current"},
                    {
                        "method_name": "math_safe_balanced",
                        "runtime_weight": 0.3,
                        "prior_weight": 0.7,
                        "stage_ratios": {"reasoning": 0.15, "verify": 0.10},
                        "refresh_intervals": {"reasoning": 128, "verify": 64},
                    },
                ],
            },
        }
        assert adaptive_griffin_methods is not None
        methods = adaptive_griffin_methods(cfg)
        self.assertEqual([method["name"] for method in methods], ["safe_dynamic_v2_current", "math_safe_balanced"])
        self.assertEqual(methods[0]["stage_ratios"]["reasoning"], 0.20)
        self.assertEqual(methods[1]["stage_ratios"]["setup"], 0.15)
        self.assertEqual(methods[1]["stage_ratios"]["reasoning"], 0.15)
        self.assertEqual(methods[1]["stage_ratios"]["verify"], 0.10)
        self.assertEqual(methods[1]["refresh_intervals"]["setup"], 128)
        self.assertEqual(methods[1]["refresh_intervals"]["reasoning"], 128)
        self.assertEqual(methods[1]["runtime_weight"], 0.3)
        self.assertEqual(methods[1]["prior_weight"], 0.7)


if __name__ == "__main__":
    unittest.main()
