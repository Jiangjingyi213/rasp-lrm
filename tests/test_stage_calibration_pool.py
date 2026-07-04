from __future__ import annotations

import unittest

from src.stage_calibration.pool import (
    allocate_source_targets,
    decontaminate,
    jaccard,
    normalize_big_math_row,
    normalize_calibration_pool_row,
    normalize_multiple_choice_row,
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
        self.assertEqual(row["answer_type"], "math")

    def test_math_solution_can_be_used_as_gold(self) -> None:
        row = normalize_big_math_row(
            {
                "problem": "Solve x+1=3.",
                "solution": "Subtract 1 from both sides to get \\boxed{2}.",
                "type": "algebra",
                "level": "Level 1",
            },
            0,
        )
        self.assertEqual(row["gold"], "Subtract 1 from both sides to get \\boxed{2}.")
        self.assertEqual(row["solution"], row["gold"])
        self.assertEqual(row["domain"], "algebra")
        self.assertEqual(row["level"], "Level 1")

    def test_allocate_source_targets_preserves_total(self) -> None:
        targets = allocate_source_targets(
            10,
            [
                {"name": "orca", "quota": 0.6},
                {"name": "openbook", "quota": 0.2},
                {"name": "commonsense", "quota": 0.2},
            ],
        )
        self.assertEqual(sum(targets), 10)
        self.assertEqual(targets, [6, 2, 2])

    def test_multiple_choice_adapter_formats_question_and_gold(self) -> None:
        row = normalize_multiple_choice_row(
            {
                "id": "q1",
                "question": "Which object is used for writing?",
                "choices": {
                    "label": ["A", "B", "C"],
                    "text": ["pen", "shoe", "cup"],
                },
                "answerKey": "A",
            },
            0,
            {"name": "commonsenseqa", "default_domain": "commonsense"},
        )
        self.assertEqual(row["gold"], "A")
        self.assertEqual(row["answer_type"], "multiple_choice")
        self.assertIn("A. pen", row["question"])
        self.assertIn(r"\boxed{}", row["question"])

    def test_normalize_calibration_pool_row_adds_source_prefix(self) -> None:
        row = normalize_calibration_pool_row(
            {
                "id": "q1",
                "question_stem": "What is true?",
                "choices": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}],
                "answerKey": "B",
            },
            0,
            {
                "name": "openbookqa",
                "name_or_path": "allenai/openbookqa",
                "adapter": "multiple_choice",
                "default_source": "openbookqa",
                "default_domain": "science_reasoning",
            },
        )
        self.assertEqual(row["id"], "openbookqa:q1")
        self.assertEqual(row["source"], "openbookqa")
        self.assertEqual(row["source_dataset"], "allenai/openbookqa")

    def test_target_adapted_math_adapter_aliases_use_math_path(self) -> None:
        gsm_row = normalize_calibration_pool_row(
            {"question": "What is 3+5?", "answer": "#### 8"},
            0,
            {
                "name": "gsm8k_train",
                "name_or_path": "gsm8k",
                "adapter": "gsm8k_train",
                "default_source": "gsm8k_train",
            },
        )
        math_row = normalize_calibration_pool_row(
            {"problem": "Compute 6/2.", "solution": "We get \\boxed{3}."},
            1,
            {
                "name": "math_train_algebra",
                "name_or_path": "EleutherAI/hendrycks_math",
                "adapter": "math_train",
                "default_source": "math_train_algebra",
            },
        )
        self.assertEqual(gsm_row["answer_type"], "math")
        self.assertEqual(gsm_row["source"], "gsm8k_train")
        self.assertEqual(math_row["answer_type"], "math")
        self.assertIn(r"\boxed{3}", math_row["gold"])


if __name__ == "__main__":
    unittest.main()
