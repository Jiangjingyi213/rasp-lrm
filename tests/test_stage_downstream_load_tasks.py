from __future__ import annotations

import unittest
from unittest.mock import patch

from src.data.load_gsm8k import (
    _normalize_arc_row,
    _normalize_bbh_row,
    load_tasks,
)


class StageDownstreamLoadTasksTest(unittest.TestCase):
    def test_normalize_arc_row_formats_multiple_choice(self) -> None:
        row = _normalize_arc_row(
            {
                "id": "arc-1",
                "question": "Which object conducts electricity?",
                "choices": {
                    "label": ["1", "2", "3"],
                    "text": ["rubber", "copper", "paper"],
                },
                "answerKey": "2",
            },
            0,
            "test",
            "arc_easy",
        )
        self.assertEqual(row["id"], "arc-1")
        self.assertEqual(row["dataset"], "arc_easy")
        self.assertEqual(row["answer_type"], "multiple_choice")
        self.assertEqual(row["gold"], "B")
        self.assertIn("A. rubber", row["question"])
        self.assertIn("B. copper", row["question"])
        self.assertIn(r"\boxed{}", row["question"])

    def test_normalize_bbh_row_preserves_exact_answers(self) -> None:
        row = _normalize_bbh_row(
            {"input": "Evaluate: not false.", "target": "True"},
            0,
            "test",
            "boolean_expressions",
        )
        self.assertEqual(row["dataset"], "bbh_selected")
        self.assertEqual(row["bbh_task"], "boolean_expressions")
        self.assertEqual(row["answer_type"], "exact")
        self.assertEqual(row["gold"], "True")
        self.assertIn(r"\boxed{}", row["question"])

    def test_normalize_bbh_row_detects_choice_answers(self) -> None:
        row = _normalize_bbh_row(
            {"input": "Choose one.\n(A) yes\n(B) no", "target": "(B)"},
            0,
            "test",
            "disambiguation_qa",
        )
        self.assertEqual(row["answer_type"], "multiple_choice")
        self.assertEqual(row["gold"], "B")

    def test_load_tasks_dispatches_arc_easy(self) -> None:
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [
                {
                    "question": "Which is a planet?",
                    "choices": {"label": ["A", "B"], "text": ["Mars", "Granite"]},
                    "answerKey": "A",
                }
            ]

        with patch("src.data.load_gsm8k.load_dataset", side_effect=fake_load_dataset):
            rows = load_tasks({"dataset": "arc_easy", "split": "test", "limit": 1})
        self.assertEqual(calls[0][0][:2], ("ai2_arc", "ARC-Easy"))
        self.assertEqual(rows[0]["gold"], "A")
        self.assertEqual(rows[0]["answer_type"], "multiple_choice")

    def test_load_tasks_dispatches_bbh_selected_with_per_task_limit(self) -> None:
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [
                {"input": f"{args[1]} item 1", "target": "yes"},
                {"input": f"{args[1]} item 2", "target": "no"},
            ]

        with patch("src.data.load_gsm8k.load_dataset", side_effect=fake_load_dataset):
            rows = load_tasks(
                {
                    "dataset": "bbh_selected",
                    "selected_tasks": ["boolean_expressions", "date_understanding"],
                    "per_task_limit": 1,
                }
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual([call[0][1] for call in calls], ["boolean_expressions", "date_understanding"])
        self.assertEqual({row["bbh_task"] for row in rows}, {"boolean_expressions", "date_understanding"})


if __name__ == "__main__":
    unittest.main()
