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

    def test_structured_prompt_forbids_closing_and_restart(self) -> None:
        prompt = build_prompt(
            "What is 1+1?",
            prompt_config={
                "explicit_stage_protocol": True,
                "forced_assistant_prefix": "<STAGE_SETUP>\n",
            },
        )
        self.assertIn("Never write closing markers such as </STAGE_SETUP>.", prompt)
        self.assertIn("Never restart from <STAGE_SETUP> after <STAGE_FINAL>.", prompt)
        self.assertIn("output only the final boxed answer", prompt)
        self.assertIn("and then stop", prompt)


if __name__ == "__main__":
    unittest.main()
