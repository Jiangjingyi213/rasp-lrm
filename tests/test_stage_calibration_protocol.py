from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.stage_calibration.protocol import (
    STAGES,
    StageTokenTracker,
    analyze_generated_ids,
    boxed_answer_complete,
    decoded_text_has_complete_stage_answer,
    marker_token_sequences,
    should_stop_after_complete_stage_answer,
)


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

    def test_setup_restart_before_final_resets_to_new_protocol_block(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 11, 2, 10, 3, 11, 4, 12, 5, 13, 6):
            tracker.feed(token)
        result = tracker.finalize()
        self.assertTrue(result["valid"])
        self.assertEqual(result["protocol_restarts"], 1)
        self.assertEqual(
            [span["stage"] for span in result["stage_spans"]],
            ["setup", "reasoning", "verify", "final"],
        )
        self.assertEqual(result["token_stages"][:5], [None, None, None, None, None])
        self.assertEqual(result["token_stages"][5:], ["setup", None, "reasoning", None, "verify", None, "final"])

    def test_missing_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 11, 2):
            tracker.feed(token)
        result = tracker.finalize()
        self.assertFalse(result["valid"])
        self.assertIn("missing_stage_markers", result["fallback_reason"])

    def test_unknown_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        result = tracker.finalize("[[STAGE_OTHER]]")
        self.assertFalse(result["valid"])
        self.assertIn("unknown_stage_marker", result["fallback_reason"])

    def test_closing_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        result = tracker.finalize("[[STAGE_SETUP]] setup </STAGE_SETUP>")
        self.assertFalse(result["valid"])
        self.assertEqual(result["fallback_reason"], "closing_stage_marker:</STAGE_SETUP>")

    def test_legacy_xml_marker_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        result = tracker.finalize("<STAGE_SETUP> setup")
        self.assertFalse(result["valid"])
        self.assertEqual(result["fallback_reason"], "legacy_stage_marker:<STAGE_SETUP>")

    def test_restarting_after_final_is_invalid(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 11, 2, 12, 3, 13, 4, 10):
            tracker.feed(token)
        result = tracker.finalize()
        self.assertFalse(result["valid"])
        self.assertEqual(result["fallback_reason"], "invalid_stage_transition:None->setup")

    def test_complete_protocol_with_boxed_answer_can_stop(self) -> None:
        tracker = StageTokenTracker(SEQUENCES)
        for token in (10, 1, 11, 2, 12, 3, 13, 4):
            tracker.feed(token)
        decoded = (
            "[[STAGE_SETUP]] setup [[STAGE_REASONING]] reason "
            "[[STAGE_VERIFY]] verify [[STAGE_FINAL]] \\boxed{\\frac{1}{2}}"
        )
        self.assertTrue(boxed_answer_complete(decoded))
        self.assertTrue(should_stop_after_complete_stage_answer(tracker, decoded))

    def test_decoded_text_complete_answer_uses_last_legal_block(self) -> None:
        decoded = (
            "[[STAGE_SETUP]] draft [[STAGE_REASONING]] restart "
            "[[STAGE_SETUP]] setup [[STAGE_REASONING]] reason "
            "[[STAGE_VERIFY]] verify [[STAGE_FINAL]] \\boxed{7}"
        )
        self.assertTrue(decoded_text_has_complete_stage_answer(decoded))

    def test_decoded_text_complete_answer_rejects_trailing_stage_label(self) -> None:
        decoded = (
            "[[STAGE_SETUP]] setup [[STAGE_REASONING]] reason "
            "[[STAGE_VERIFY]] verify [[STAGE_FINAL]] \\boxed{7} [[STAGE_SETUP]]"
        )
        self.assertFalse(decoded_text_has_complete_stage_answer(decoded))

    def test_transition_occurs_only_after_complete_multitoken_marker(self) -> None:
        sequences = dict(SEQUENCES)
        sequences["setup"] = (20, 21)
        tracker = StageTokenTracker(sequences)
        self.assertIsNone(tracker.feed(20))
        self.assertEqual(tracker.feed(21), "setup")
        self.assertEqual(tracker.transitions[0]["generated_tokens"], 2)

    def test_decoded_text_fallback_handles_contextual_marker_tokens(self) -> None:
        class FakeTokenizer:
            chunks = [
                "`",
                "[[STAGE_SETUP]]",
                "` setup ",
                "[[STAGE_REASONING]]",
                " reason ",
                "[[STAGE_VERIFY]]",
                " verify ",
                "[[STAGE_FINAL]]",
                " \\boxed{1}",
            ]

            def __call__(self, _text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[999])

            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.chunks[: len(ids)])

        result = analyze_generated_ids(FakeTokenizer(), list(range(9)))
        self.assertTrue(result["valid"])
        self.assertEqual(result["detected_by"], "decoded_text")
        self.assertEqual(result["stage_spans"][-1]["stage"], "final")

    def test_decoded_text_fallback_rejects_closing_marker(self) -> None:
        class FakeTokenizer:
            chunks = [
                "[[STAGE_SETUP]]",
                " setup ",
                "</STAGE_SETUP>",
                "[[STAGE_REASONING]]",
                " reason ",
                "[[STAGE_VERIFY]]",
                " verify ",
                "[[STAGE_FINAL]]",
                " \\boxed{1}",
            ]

            def __call__(self, _text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[999])

            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.chunks[: len(ids)])

        result = analyze_generated_ids(FakeTokenizer(), list(range(9)))
        self.assertFalse(result["valid"])
        self.assertEqual(result["fallback_reason"], "closing_stage_marker:</STAGE_SETUP>")

    def test_decoded_text_fallback_uses_last_complete_block(self) -> None:
        class FakeTokenizer:
            chunks = [
                "[[STAGE_SETUP]]",
                " old ",
                "[[STAGE_REASONING]]",
                " draft ",
                "[[STAGE_SETUP]]",
                " setup ",
                "[[STAGE_REASONING]]",
                " reason ",
                "[[STAGE_VERIFY]]",
                " verify ",
                "[[STAGE_FINAL]]",
                " \\boxed{1}",
            ]

            def __call__(self, _text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[999])

            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.chunks[: len(ids)])

        result = analyze_generated_ids(FakeTokenizer(), list(range(12)))
        self.assertTrue(result["valid"])
        self.assertEqual(result["detected_by"], "decoded_text")
        self.assertEqual(result["leading_stage_labels_ignored"], 2)
        self.assertEqual(result["stage_spans"][0]["stage"], "setup")

    def test_marker_token_sequences_include_trailing_newline_variant(self) -> None:
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                table = {
                    "[[STAGE_SETUP]]": [10],
                    "[[STAGE_SETUP]]\n": [110],
                    "[[STAGE_REASONING]]": [11],
                    "[[STAGE_VERIFY]]": [12],
                    "[[STAGE_FINAL]]": [13],
                }
                return SimpleNamespace(input_ids=table.get(text, [999]))

        sequences = marker_token_sequences(FakeTokenizer())
        tracker = StageTokenTracker(sequences)
        self.assertEqual(tracker.feed(110), "setup")


if __name__ == "__main__":
    unittest.main()
