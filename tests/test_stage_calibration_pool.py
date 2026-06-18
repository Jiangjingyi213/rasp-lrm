from __future__ import annotations

import unittest

from src.stage_calibration.pool import (
    decontaminate,
    jaccard,
    normalize_text,
    source_allowed,
    stratified_split,
)


class StageCalibrationPoolTest(unittest.TestCase):
    def test_source_allowlist_excludes_math(self) -> None:
        self.assertTrue(source_allowed("HARP"))
        self.assertFalse(source_allowed("MATH"))
        self.assertFalse(source_allowed("unknown"))

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


if __name__ == "__main__":
    unittest.main()
