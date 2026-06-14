from __future__ import annotations

import unittest

from src.rasp.budget_controller import FixedSingleWindowController, RuntimeObservation
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


if __name__ == "__main__":
    unittest.main()
