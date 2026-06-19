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

    def test_symbolic_gold_does_not_fall_back_to_last_number(self) -> None:
        self.assertEqual(extract_answer(r"4\sqrt{2}"), r"4 sqrt{2}")
        self.assertFalse(answer_match(r"\boxed{2}", r"4\sqrt{2}", use_math_verify=False))

    def test_optional_math_verifier_handles_symbolic_fallback(self) -> None:
        with (
            patch("src.metrics.answer_match.math_verify_parse", side_effect=lambda value: [value]),
            patch("src.metrics.answer_match.math_verify_verify", return_value=True),
        ):
            self.assertTrue(answer_match(r"\boxed{x^2-1}", "(x-1)(x+1)"))

    def test_math_verify_uses_extracted_prediction_not_full_reasoning(self) -> None:
        calls = []

        def fake_parse(value):
            calls.append(value)
            return [value]

        def fake_verify(gold, prediction):
            return gold == prediction

        with (
            patch("src.metrics.answer_match.math_verify_parse", side_effect=fake_parse),
            patch("src.metrics.answer_match.math_verify_verify", side_effect=fake_verify),
        ):
            self.assertFalse(
                answer_match(
                    "Reasoning mentions 4\\sqrt{2}. Therefore \\boxed{2}.",
                    "4\\sqrt{2}",
                )
            )
        self.assertNotIn("Reasoning mentions 4\\sqrt{2}. Therefore \\boxed{2}.", calls)


if __name__ == "__main__":
    unittest.main()
