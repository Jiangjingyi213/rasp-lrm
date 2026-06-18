from __future__ import annotations

import unittest

from src.stage_calibration.protocol import STAGES, StageTokenTracker


SEQUENCES = {stage: (index + 10,) for index, stage in enumerate(STAGES)}


class StageCalibrationProtocolTest(unittest.TestCase):
    def test_valid_protocol_assigns_only_content_tokens(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 2, 11, 3, 12, 4, 13, 5):
            tracker.feed(token)
        result = tracker.finalize()
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["token_stages"],
            [None, "setup", "setup", None, "reasoning", None, "verify", None, "final"],
        )

    def test_duplicate_or_out_of_order_marker_falls_back_dense(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        tracker.feed(10)
        tracker.feed(12)
        result = tracker.finalize()
        self.assertFalse(result["valid"])
        self.assertIn("invalid_stage_transition", result["fallback_reason"])

    def test_missing_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 11, 2):
            tracker.feed(token)
        result = tracker.finalize()
        self.assertFalse(result["valid"])
        self.assertIn("missing_stage_markers", result["fallback_reason"])

    def test_unknown_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        result = tracker.finalize("<STAGE_OTHER>")
        self.assertFalse(result["valid"])
        self.assertIn("unknown_stage_marker", result["fallback_reason"])

    def test_transition_occurs_only_after_complete_multitoken_marker(self) -> None:
        sequences = dict(SEQUENCES)
        sequences["setup"] = (20, 21)
        tracker = StageTokenTracker(sequences)
        self.assertIsNone(tracker.feed(20))
        self.assertEqual(tracker.feed(21), "setup")
        self.assertEqual(tracker.transitions[0]["generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
