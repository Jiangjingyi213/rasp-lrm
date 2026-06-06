from __future__ import annotations

import unittest

from src.rasp.safe_oracle import (
    allocate_budget_aware_safe_oracle,
    allocate_causal_budget_aware_safe_oracle,
    build_safe_oracle_steps,
)


class SafeOracleTest(unittest.TestCase):
    def test_max_and_monotonic_safe_ratios_are_both_recorded(self) -> None:
        rows = [
            {"dataset": "gsm8k", "id": "x", "segment_id": 0, "ratio": 0.0, "flipped": False},
            {"dataset": "gsm8k", "id": "x", "segment_id": 0, "ratio": 0.1, "flipped": True},
            {"dataset": "gsm8k", "id": "x", "segment_id": 0, "ratio": 0.2, "flipped": False},
        ]
        [summary] = build_safe_oracle_steps(rows)
        self.assertEqual(summary["max_safe_ratio"], 0.2)
        self.assertEqual(summary["monotonic_safe_ratio"], 0.0)
        self.assertTrue(summary["non_monotonic"])

    def test_budget_oracle_never_exceeds_safe_caps_or_total_budget(self) -> None:
        steps = [
            {"id": "a", "tested_ratios": [0.0, 0.1, 0.2, 0.4], "max_safe_ratio": 0.4},
            {"id": "b", "tested_ratios": [0.0, 0.1, 0.2, 0.4], "max_safe_ratio": 0.1},
        ]
        selected = allocate_budget_aware_safe_oracle(steps, 0.2)
        self.assertLessEqual(sum(row["selected_ratio"] for row in selected), 0.4)
        self.assertTrue(all(row["selected_ratio"] <= row["max_safe_ratio"] for row in selected))

    def test_causal_oracle_is_problem_local_and_prefix_budget_safe(self) -> None:
        steps = [
            {
                "dataset": "gsm8k",
                "id": "a",
                "segment_id": 0,
                "segment_index": 0,
                "tested_ratios": [0.0, 0.1, 0.2, 0.4],
                "monotonic_safe_ratio": 0.4,
            },
            {
                "dataset": "gsm8k",
                "id": "a",
                "segment_id": 1,
                "segment_index": 1,
                "tested_ratios": [0.0, 0.1, 0.2, 0.4],
                "monotonic_safe_ratio": 0.4,
            },
            {
                "dataset": "gsm8k",
                "id": "b",
                "segment_id": 0,
                "segment_index": 0,
                "tested_ratios": [0.0, 0.1, 0.2, 0.4],
                "monotonic_safe_ratio": 0.4,
            },
        ]
        selected = allocate_causal_budget_aware_safe_oracle(steps, 0.15)
        self.assertEqual([row["selected_ratio"] for row in selected], [0.1, 0.2, 0.1])
        self.assertEqual(selected[0]["available_budget_before_selection"], 0.15)
        self.assertAlmostEqual(selected[1]["available_budget_before_selection"], 0.2)
        self.assertEqual(selected[2]["available_budget_before_selection"], 0.15)
        self.assertTrue(all(row["prefix_average_after_selection"] <= 0.15 + 1e-9 for row in selected))


if __name__ == "__main__":
    unittest.main()
