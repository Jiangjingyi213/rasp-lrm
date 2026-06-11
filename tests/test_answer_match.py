from __future__ import annotations

import unittest

from src.metrics.answer_match import answer_match


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


if __name__ == "__main__":
    unittest.main()
