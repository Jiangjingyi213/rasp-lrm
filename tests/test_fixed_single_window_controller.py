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


if __name__ == "__main__":
    unittest.main()
