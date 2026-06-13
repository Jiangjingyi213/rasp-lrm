from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.main_train_action_risk_controller import simulate, threshold_grid
from src.rasp.action_risk_single_window import (
    ACTION_RISK_CHECKPOINT_SCHEMA,
    CONTEXT_FEATURE_NAMES,
    ActionRiskSingleWindowController,
    monotonic_risk_envelope,
)
from src.rasp.budget_controller import RuntimeObservation


class IdentityTransform:
    def transform(self, values):
        return np.asarray(values)


class FirstColumnTransform:
    def transform(self, values):
        return np.asarray(values)[:, :1]


class RatioRiskModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def predict_proba(self, values):
        values = np.asarray(values)
        risks = np.clip(values[:, -2] + self.offset, 0.0, 1.0)
        return np.stack([1.0 - risks, risks], axis=1)


class HiddenRiskModel:
    def predict_proba(self, values):
        values = np.asarray(values)
        risks = np.clip(values[:, 0], 0.0, 1.0)
        return np.stack([1.0 - risks, risks], axis=1)


class ActionRiskSingleWindowTest(unittest.TestCase):
    def checkpoint(self, root: Path, *, max_new_tokens: int = 768) -> Path:
        path = root / "policy.joblib"
        joblib.dump(
            {
                "schema": ACTION_RISK_CHECKPOINT_SCHEMA,
                "context_feature_names": CONTEXT_FEATURE_NAMES,
                "max_new_tokens": max_new_tokens,
                "window_tokens": 16,
                "eligible_boundaries": (32, 96, 160),
                "candidate_ratios": (0.05, 0.10, 0.20, 0.30),
                "context_scaler": IdentityTransform(),
                "context_model": RatioRiskModel(),
                "hidden_scaler": IdentityTransform(),
                "hidden_pca": FirstColumnTransform(),
                "hidden_model": HiddenRiskModel(),
                "operating_points": {
                    "balanced": {
                        "eligible": True,
                        "context_risk_threshold": 0.20,
                        "hidden_veto_eligible": True,
                        "hidden_veto_threshold": 0.15,
                    }
                },
            },
            path,
        )
        return path

    def test_selects_once_at_eligible_boundary_then_stays_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ActionRiskSingleWindowController(
                checkpoint_path=str(self.checkpoint(Path(directory))),
                policy_variant="context_only",
                operating_point="balanced",
                max_new_tokens=768,
            )
            ratios = [
                controller.choose_ratio(RuntimeObservation(tokens, 0.2, 0.8, np.asarray([[0.1]])))
                for tokens in (0, 16, 32, 48, 96)
            ]
        self.assertEqual(ratios, [0.0, 0.0, 0.20, 0.0, 0.0])

    def test_hidden_can_only_veto(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ActionRiskSingleWindowController(
                checkpoint_path=str(self.checkpoint(Path(directory))),
                policy_variant="hidden_veto",
                operating_point="balanced",
                max_new_tokens=768,
            )
            selected = controller.choose_ratio(
                RuntimeObservation(32, 0.2, 0.8, np.asarray([[0.25]]))
            )
        self.assertEqual(selected, 0.0)

    def test_missing_hidden_falls_back_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ActionRiskSingleWindowController(
                checkpoint_path=str(self.checkpoint(Path(directory))),
                policy_variant="hidden_veto",
                operating_point="balanced",
                max_new_tokens=768,
            )
            self.assertEqual(controller.choose_ratio(RuntimeObservation(32, 0.2, 0.8)), 0.0)
            self.assertEqual(controller.last_decision["reason"], "missing_hidden_state")

    def test_rejects_generation_limit_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ActionRiskSingleWindowController(
                    checkpoint_path=str(self.checkpoint(Path(directory))),
                    policy_variant="context_only",
                    operating_point="balanced",
                    max_new_tokens=512,
                )

    def test_rejects_runtime_ratio_grid_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ActionRiskSingleWindowController(
                    checkpoint_path=str(self.checkpoint(Path(directory))),
                    policy_variant="context_only",
                    operating_point="balanced",
                    max_new_tokens=768,
                    runtime_ratios=(0.0, 0.05, 0.10),
                )

    def test_monotonic_envelope_blocks_later_risk_inversion(self) -> None:
        self.assertEqual(monotonic_risk_envelope(np.asarray([0.1, 0.4, 0.2])).tolist(), [0.1, 0.4, 0.4])

    def test_calibration_simulation_uses_first_accepted_boundary(self) -> None:
        rows = []
        for boundary, flipped in ((32, True), (96, False)):
            rows.append(
                {
                    "dataset": "gsm8k",
                    "id": "one",
                    "generated_tokens_at_boundary": boundary,
                    "ratio": 0.10,
                    "flipped": flipped,
                    "causal_context_action_risk": 0.10,
                    "hidden_context_action_risk": 0.10,
                }
            )
        metrics = simulate(rows, context_threshold=0.20)
        self.assertEqual(metrics["gsm8k"]["selected"], 1)
        self.assertEqual(metrics["gsm8k"]["problem_flip_rate"], 1.0)

    def test_calibration_counts_problem_without_eligible_boundary_as_dense(self) -> None:
        rows = [
            {
                "dataset": "gsm8k",
                "id": "eligible",
                "generated_tokens_at_boundary": 32,
                "ratio": 0.10,
                "flipped": False,
                "causal_context_action_risk": 0.10,
                "hidden_context_action_risk": 0.10,
            },
            {
                "dataset": "gsm8k",
                "id": "no_exact_boundary",
                "generated_tokens_at_boundary": 48,
                "ratio": 0.10,
                "flipped": False,
                "causal_context_action_risk": 0.10,
                "hidden_context_action_risk": 0.10,
            },
        ]
        metrics = simulate(rows, context_threshold=0.20)
        self.assertEqual(metrics["gsm8k"]["problems"], 2)
        self.assertEqual(metrics["gsm8k"]["selected"], 1)
        self.assertEqual(metrics["gsm8k"]["action_coverage"], 0.5)
        self.assertEqual(metrics["gsm8k"]["average_action_ratio"], 0.05)

    def test_threshold_grid_is_bounded_and_spans_range(self) -> None:
        grid = threshold_grid([float(value) for value in range(1000)], maximum_points=21)
        self.assertEqual(len(grid), 21)
        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 999.0)


if __name__ == "__main__":
    unittest.main()
