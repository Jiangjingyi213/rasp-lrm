from __future__ import annotations

import unittest

from src.main_offline_baselines import group_steps
from src.main_rasp_zero_offline import DEFAULT_STAGE_CAPS, select_budget_policy


def action_rows(item_id: str, segment_id: int, segment_type: str) -> list[dict]:
    rows = []
    for ratio in (0.2, 0.4, 0.6):
        rows.append(
            {
                "dataset": "fake",
                "id": item_id,
                "segment_id": segment_id,
                "segment_type": segment_type,
                "module": "mlp_block",
                "ratio": ratio,
                "pruned_layers": [4],
                "flipped": ratio > 0.4,
            }
        )
    return rows


class RaspZeroOfflineTest(unittest.TestCase):
    def test_budget_allocation_prioritizes_low_risk_and_respects_stage_caps(self) -> None:
        rows = [
            *action_rows("p0", 0, "understanding"),
            *action_rows("p0", 1, "verification"),
            *action_rows("p1", 0, "derivation"),
        ]
        steps = group_steps(rows)
        scores = {
            ("fake", "p0", 0): 0.1,
            ("fake", "p0", 1): 0.2,
            ("fake", "p1", 0): 0.9,
        }
        selected = select_budget_policy(
            steps,
            scores,
            module="mlp_block",
            ratios=[0.2, 0.4, 0.6],
            target_average_ratio=0.4,
            stage_caps=DEFAULT_STAGE_CAPS,
        )
        selected_by_step = {(row["id"], row["segment_id"]): row for row in selected}
        self.assertEqual(selected_by_step[("p0", 0)]["ratio"], 0.6)
        self.assertEqual(selected_by_step[("p0", 1)]["ratio"], 0.2)
        self.assertEqual(selected_by_step[("p1", 0)]["ratio"], 0.4)
        self.assertLessEqual(sum(float(row["ratio"]) for row in selected) / len(selected), 0.4 + 1e-9)


if __name__ == "__main__":
    unittest.main()
