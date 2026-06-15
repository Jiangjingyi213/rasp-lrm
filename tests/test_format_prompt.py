from __future__ import annotations

import unittest

from src.data.format_prompt import build_prompt


class FormatPromptTest(unittest.TestCase):
    def test_boxed_answer_instruction(self) -> None:
        prompt = build_prompt("What is 1+1?", prompt_config={"answer_format": "boxed"})
        self.assertIn(r"Put the final answer in \boxed{}.", prompt)

    def test_default_instruction_remains_backward_compatible(self) -> None:
        prompt = build_prompt("What is 1+1?")
        self.assertIn("Final answer:", prompt)


if __name__ == "__main__":
    unittest.main()
