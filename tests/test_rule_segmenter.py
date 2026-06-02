from __future__ import annotations

import unittest

from src.segmentation.rule_segmenter import segment_text


class RuleSegmenterTest(unittest.TestCase):
    def test_short_final_answer_is_not_merged_with_short_previous_segment(self) -> None:
        segments = segment_text("Work.\n\nFinal answer: 3", min_chars=24)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1]["segment_type"], "final")
        self.assertEqual(segments[-1]["text"], "Final answer: 3")


if __name__ == "__main__":
    unittest.main()
