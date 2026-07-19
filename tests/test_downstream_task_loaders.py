from __future__ import annotations

import unittest
from unittest.mock import patch

from src.data import load_gsm8k
from src.data.load_gsm8k import (
    load_competition_hf,
    _normalize_competition_row,
    _normalize_gpqa_row,
    load_tasks,
)


class DownstreamTaskLoadersTest(unittest.TestCase):
    def test_gpqa_diamond_adapter_formats_multiple_choice(self) -> None:
        row = _normalize_gpqa_row(
            {
                "Record ID": "gpqa-1",
                "Question": "Which statement is correct?",
                "Correct Answer": "Correct option",
                "Incorrect Answer 1": "Wrong one",
                "Incorrect Answer 2": "Wrong two",
                "Incorrect Answer 3": "Wrong three",
            },
            0,
            "train",
            seed=1,
        )
        self.assertEqual(row["dataset"], "gpqa_diamond")
        self.assertEqual(row["answer_type"], "multiple_choice")
        self.assertRegex(row["gold"], r"^[A-D]$")
        self.assertIn("Options:", row["question"])
        self.assertIn(r"\boxed{}", row["question"])
        self.assertEqual(len(row["choices"]), 4)

    def test_competition_adapter_handles_dapo_reward_model(self) -> None:
        row = _normalize_competition_row(
            {
                "prompt": [{"role": "user", "content": "Find x if x+1=3."}],
                "reward_model": {"ground_truth": "2"},
                "data_source": "aime2025",
            },
            0,
            "train",
            "aime2025",
        )
        self.assertEqual(row["dataset"], "aime2025")
        self.assertEqual(row["gold"], "2")
        self.assertEqual(row["answer_type"], "math")
        self.assertIn(r"\boxed{}", row["question"])

    def test_competition_adapter_handles_amc_choice_answer(self) -> None:
        row = _normalize_competition_row(
            {
                "question": "Which value is prime?\nA. 4\nB. 5\nC. 6",
                "answer": "B",
            },
            0,
            "train",
            "amc2023",
        )
        self.assertEqual(row["dataset"], "amc2023")
        self.assertEqual(row["gold"], "B")
        self.assertEqual(row["answer_type"], "multiple_choice")
        self.assertIn("option letter", row["question"])

    def test_load_tasks_supports_new_dataset_names(self) -> None:
        fake_rows = [
            {
                "Question": "Which option?",
                "Correct Answer": "yes",
                "Incorrect Answer 1": "no",
                "Incorrect Answer 2": "maybe",
                "Incorrect Answer 3": "never",
            }
        ]
        with patch.object(load_gsm8k, "load_dataset", return_value=fake_rows) as mocked:
            rows = load_tasks({"dataset": "gpqa_diamond"})
        self.assertEqual(rows[0]["dataset"], "gpqa_diamond")
        mocked.assert_called_once()

        competition_rows = [{"question": "Compute 1+1.", "answer": "2"}]
        with patch.object(load_gsm8k, "load_dataset", return_value=competition_rows):
            self.assertEqual(load_tasks({"dataset": "aime2025"})[0]["dataset"], "aime2025")
        with patch.object(load_gsm8k, "load_dataset", return_value=competition_rows):
            self.assertEqual(load_tasks({"dataset": "amc2023"})[0]["dataset"], "amc2023")

    def test_competition_hf_falls_back_when_split_is_missing(self) -> None:
        competition_rows = [{"question": "Compute 1+1.", "answer": "2"}]

        def fake_load_dataset(*args, split):
            if split == "train":
                raise ValueError('Unknown split "train". Should be one of [\'test\'].')
            self.assertEqual(split, "test")
            return competition_rows

        with patch.object(load_gsm8k, "load_dataset", side_effect=fake_load_dataset):
            rows = load_competition_hf(
                {"name_or_path": "math-ai/aime25", "split": "train"},
                "aime2025",
                "math-ai/aime25",
            )
        self.assertEqual(rows[0]["id"], "aime2025-test-0")
        self.assertEqual(rows[0]["gold"], "2")


if __name__ == "__main__":
    unittest.main()
