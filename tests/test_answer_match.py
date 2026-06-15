from __future__ import annotations

import unittest
from unittest.mock import patch

from src.metrics.answer_match import answer_match, extract_answer


class AnswerMatchTest(unittest.TestCase):
    def test_single_word_categorical_gold_matches_final_answer_sentence(self) -> None:
        self.assertTrue(
            answer_match(
                "Final answer: Evelyn has the greatest average speed.",
                r"\text{Evelyn}",
            )
        )

    def test_single_word_categorical_gold_requires_whole_word(self) -> None:
        self.assertFalse(answer_match("Final answer: Evelynne", r"\text{Evelyn}"))

    def test_extracts_nested_boxed_answer(self) -> None:
        self.assertEqual(extract_answer(r"Therefore \boxed{\frac{14}{3}}."), "14/3")

    def test_fraction_and_decimal_are_equivalent_without_optional_grader(self) -> None:
        self.assertTrue(answer_match(r"\boxed{\frac{1}{2}}", "0.5", use_math_verify=False))

    def test_optional_math_verifier_handles_symbolic_fallback(self) -> None:
        with (
            patch("src.metrics.answer_match.math_verify_parse", side_effect=lambda value: [value]),
            patch("src.metrics.answer_match.math_verify_verify", return_value=True),
        ):
            self.assertTrue(answer_match(r"\boxed{x^2-1}", "(x-1)(x+1)"))


if __name__ == "__main__":
    unittest.main()
