from __future__ import annotations

import unittest

from src.rasp.budget_controller import (
    FixedMultiWindowController,
    FixedSingleWindowController,
    RuntimeObservation,
)
from src.rasp.on_policy import prior_action_candidate_events
from src.rasp.window_sampling import boundary_positions


class FixedSingleWindowControllerTest(unittest.TestCase):
    def test_activates_once_then_restores_dense(self) -> None:
        controller = FixedSingleWindowController(boundary_tokens=32, ratio=0.4, window_tokens=16)
        ratios = [
            controller.choose_ratio(RuntimeObservation(tokens, 0.0, 1.0))
            for tokens in (0, 16, 32, 48, 64)
        ]
        self.assertEqual(ratios, [0.0, 0.0, 0.4, 0.0, 0.0])

    def test_rejects_unaligned_boundary(self) -> None:
        with self.assertRaises(ValueError):
            FixedSingleWindowController(boundary_tokens=33, ratio=0.4, window_tokens=16)

    def test_uniform_boundary_sampling_spans_full_trajectory(self) -> None:
        self.assertEqual(
            boundary_positions(160, 16, 4, "uniform_full_trajectory"),
            [0, 48, 96, 144],
        )

    def test_explicit_boundary_sampling_filters_short_trajectories(self) -> None:
        self.assertEqual(
            boundary_positions(120, 16, None, explicit_positions=[32, 96, 160]),
            [32, 96],
        )

    def test_explicit_boundary_sampling_requires_alignment(self) -> None:
        with self.assertRaises(ValueError):
            boundary_positions(200, 16, None, explicit_positions=[32, 95])

    def test_causal_grid_covers_trajectory_and_isolates_tail_anchor(self) -> None:
        self.assertEqual(
            boundary_positions(
                160,
                16,
                None,
                "causal_grid",
                decision_start=32,
                decision_stride=32,
            ),
            [32, 64, 96, 128],
        )
        self.assertEqual(
            boundary_positions(
                160,
                16,
                None,
                "causal_grid",
                decision_start=32,
                decision_stride=32,
                include_tail_anchor=True,
            ),
            [32, 64, 96, 128, 144],
        )

    def test_fixed_multi_window_respects_cooldown_limit_and_reset(self) -> None:
        controller = FixedMultiWindowController(ratio=0.2, cadence_tokens=32, max_windows=2)
        values = [
            controller.choose_ratio(RuntimeObservation(tokens, 0.0, 1.0))
            for tokens in (0, 16, 32, 48, 64, 80, 96)
        ]
        self.assertEqual(values, [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0])
        self.assertEqual(len(controller.action_history), 2)
        self.assertEqual(controller.last_decision["cumulative_pruned_token_ratio_mass"], 6.4)
        controller.reset()
        self.assertEqual(
            controller.choose_ratio(RuntimeObservation(32, 0.0, 1.0)),
            0.2,
        )

    def test_fixed_multi_window_falls_back_dense_on_nonfinite_observation(self) -> None:
        controller = FixedMultiWindowController(ratio=0.2, cadence_tokens=32, max_windows=2)
        self.assertEqual(
            controller.choose_ratio(RuntimeObservation(32, float("nan"), 1.0)),
            0.0,
        )
        self.assertEqual(controller.last_decision["reason"], "non_finite_observation")
        self.assertEqual(
            controller.choose_ratio(RuntimeObservation(64, 0.0, 1.0)),
            0.2,
        )

    def test_on_policy_candidates_require_an_earlier_action(self) -> None:
        events = [
            {
                "generated_tokens": boundary,
                "selected_ratio": 0.2 if boundary == 32 else 0.0,
                "hidden_state": [float(boundary)],
                "decision": {"window_activated": boundary == 32},
            }
            for boundary in (0, 16, 32, 48, 64, 80, 96)
        ]
        candidates = prior_action_candidate_events(
            events,
            token_count=128,
            window_tokens=16,
            cooldown_tokens=16,
            decision_start=32,
            decision_stride=32,
            max_boundaries=2,
        )
        self.assertEqual([row["generated_tokens"] for row in candidates], [64, 96])


if __name__ == "__main__":
    unittest.main()
