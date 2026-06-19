from __future__ import annotations

import unittest

from src.stage_calibration.pool import (
    decontaminate,
    jaccard,
    normalize_big_math_row,
    normalize_text,
    source_allowed,
    strip_embedded_answer,
    stratified_split,
)


class StageCalibrationPoolTest(unittest.TestCase):
    def test_source_allowlist_excludes_math(self) -> None:
        self.assertTrue(source_allowed("HARP"))
        self.assertTrue(source_allowed("Orca-Math"))
        self.assertFalse(source_allowed("MATH"))
        self.assertFalse(source_allowed("unknown"))
        self.assertTrue(source_allowed("olympiads", allowed=("*",), excluded=("MATH",)))
        self.assertFalse(source_allowed("synthetic-gsm8k", allowed=("*",), excluded=("*gsm8k*",)))

    def test_exact_and_near_test_overlap_are_removed(self) -> None:
        protected = [{"id": "test-1", "question": "Compute the value of 2 + 2."}]
        kept, excluded = decontaminate(
            [
                {"id": "a", "question": "Compute the value of 2 + 2."},
                {"id": "b", "question": "Find the derivative of x squared."},
            ],
            protected,
        )
        self.assertEqual([row["id"] for row in kept], ["b"])
        self.assertEqual(excluded[0]["exclusion_reason"], "exact_test_overlap")

    def test_split_is_disjoint(self) -> None:
        rows = [
            {"id": str(index), "source": "HARP", "domain": "algebra", "question": str(index)}
            for index in range(10)
        ]
        calibration, dev = stratified_split(rows, 4, 3, seed=1)
        self.assertFalse({row["id"] for row in calibration} & {row["id"] for row in dev})

    def test_normalization_and_jaccard(self) -> None:
        self.assertEqual(normalize_text("Value: 2+2"), "value 2 2")
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_embedded_answer_is_removed_from_question(self) -> None:
        self.assertEqual(strip_embedded_answer("Compute 1+1.\n\nAnswer: 2."), "Compute 1+1.")
        row = normalize_big_math_row(
            {"question": "Compute 2+2.\nAnswer: 4", "answer": "4", "source": "olympiads"},
            0,
        )
        self.assertEqual(row["question"], "Compute 2+2.")
        self.assertEqual(row["gold"], "4")


if __name__ == "__main__":
    unittest.main()
