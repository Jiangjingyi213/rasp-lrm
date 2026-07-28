from __future__ import annotations

import unittest
from unittest.mock import patch

from src.data import load_gsm8k
from src.data.load_gsm8k import (
    _normalize_jeebench_row,
    _normalize_livecodebench_row,
    load_competition_hf,
    load_gpqa_hf,
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

    def test_livecodebench_adapter_formats_code_generation(self) -> None:
        row = _normalize_livecodebench_row(
            {
                "question_id": "lcb-1",
                "question_title": "Two Sum",
                "question_content": "Read two integers and print their sum.",
                "starter_code": "def solve():\n    pass",
                "public_test_cases": [{"input": "1 2\n", "output": "3\n"}],
            },
            0,
            "test",
        )
        self.assertEqual(row["dataset"], "livecodebench")
        self.assertEqual(row["answer_type"], "code_generation")
        self.assertIn("Starter code:", row["question"])
        self.assertTrue(row["gold"].startswith("__LCB_TESTS__"))

    def test_jeebench_adapter_handles_multiple_select(self) -> None:
        row = _normalize_jeebench_row(
            {
                "id": "jee-1",
                "question": "Which statements are true?",
                "options": {"A": "first", "B": "second", "C": "third"},
                "answer": "CA",
                "question_type": "multiple correct",
            },
            0,
            "test",
        )
        self.assertEqual(row["dataset"], "jeebench")
        self.assertEqual(row["gold"], "AC")
        self.assertEqual(row["answer_type"], "multiple_select")
        self.assertIn("alphabetical order", row["question"])

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
        lcb_rows = [{"question_content": "Write solve().", "canonical_solution": "def solve(): pass"}]
        with patch.object(load_gsm8k, "load_dataset", return_value=lcb_rows):
            self.assertEqual(load_tasks({"dataset": "livecodebench"})[0]["dataset"], "livecodebench")
        jee_rows = [{"question": "Compute 1+1.", "answer": "2"}]
        with patch.object(load_gsm8k, "load_dataset", return_value=jee_rows):
            self.assertEqual(load_tasks({"dataset": "jeebench"})[0]["dataset"], "jeebench")

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

    def test_gpqa_hf_falls_back_to_public_simple_evals_mirror(self) -> None:
        gpqa_rows = [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "Right",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        ]
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ("Idavidrein/gpqa", "gpqa_diamond"):
                raise FileNotFoundError("official GPQA source is unavailable")
            self.assertEqual(args, ("zai-org/glm-simple-evals-dataset",))
            self.assertEqual(kwargs["data_files"], "gpqa/gpqa_diamond.csv")
            return gpqa_rows

        with patch.object(load_gsm8k, "load_dataset", side_effect=fake_load_dataset):
            rows = load_gpqa_hf(
                {
                    "name_or_path": "Idavidrein/gpqa",
                    "dataset_config": "gpqa_diamond",
                    "split": "train",
                }
            )
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(rows[0]["dataset"], "gpqa_diamond")
        self.assertEqual(rows[0]["answer_type"], "multiple_choice")


if __name__ == "__main__":
    unittest.main()
