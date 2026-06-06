from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.probes.rasp_train_dataset import DEFAULT_RATIOS, RaspTrainPolicyDataset, ratio_index
from src.main_train_rasp_train_router import problem_level_three_way_split
from src.rasp.budget_controller import RuntimeObservation
from src.rasp.train_controller import RaspTrainPolicyController
from src.rasp.train_policy import POLICY_FEATURE_SCHEMA, ActionRiskPolicyNet, action_risk_loss
from src.utils.io import write_jsonl


class RaspTrainPolicyTest(unittest.TestCase):
    def test_ratio_index_requires_supported_ratio(self) -> None:
        self.assertEqual(ratio_index(0.2, DEFAULT_RATIOS), DEFAULT_RATIOS.index(0.2))
        with self.assertRaises(ValueError):
            ratio_index(0.15, DEFAULT_RATIOS)

    def test_policy_dataset_builds_one_step_budget_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "dataset": "gsm8k",
                    "id": "x",
                    "segment_id": 0,
                    "entropy": 0.1,
                    "confidence": 0.9,
                    "position": 0.0,
                    "target_budget": 0.2,
                    "available_budget_before_selection": 0.2,
                    "oracle_ratio": 0.1,
                    "candidate_flipped": [False, False, False, False, True, True, True],
                    "candidate_unsafe": [False, False, False, False, True, True, True],
                }
            ]
            write_jsonl(root / "rows.jsonl", rows)
            torch.save(torch.zeros(1, 4), root / "hidden.pt")
            dataset = RaspTrainPolicyDataset(root / "rows.jsonl", root / "hidden.pt")
            features, unsafe_mask, flipped_mask, ratios, target_budget, index = dataset[0]
            self.assertEqual(index, 0)
            self.assertAlmostEqual(float(target_budget), 0.2)
            self.assertEqual(tuple(unsafe_mask.shape), (len(DEFAULT_RATIOS),))
            self.assertEqual(tuple(flipped_mask.shape), (len(DEFAULT_RATIOS),))
            self.assertEqual(tuple(ratios.shape), (len(DEFAULT_RATIOS),))
            self.assertGreater(features.numel(), 4)

    def test_three_way_split_keeps_problems_disjoint(self) -> None:
        rows = [
            {"dataset": "gsm8k", "id": str(problem), "segment_id": segment}
            for problem in range(10)
            for segment in range(2)
        ]
        train, calibration, test, _keys = problem_level_three_way_split(rows, 0.4, 1)
        problem_sets = [
            {(rows[index]["dataset"], rows[index]["id"]) for index in indices}
            for indices in (train, calibration, test)
        ]
        self.assertTrue(problem_sets[0].isdisjoint(problem_sets[1]))
        self.assertTrue(problem_sets[0].isdisjoint(problem_sets[2]))
        self.assertTrue(problem_sets[1].isdisjoint(problem_sets[2]))
        repeated = problem_level_three_way_split(rows, 0.4, 1)
        self.assertEqual((train, calibration, test), repeated[:3])

    def test_action_risk_loss_is_finite_and_rewards_monotonic_risk(self) -> None:
        logits = torch.zeros(2, len(DEFAULT_RATIOS))
        unsafe_mask = torch.zeros(2, len(DEFAULT_RATIOS))
        unsafe_mask[:, 4:] = 1.0
        loss = action_risk_loss(
            logits,
            unsafe_mask,
            positive_weight=2.0,
            monotonic_weight=1.0,
            ranking_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_runtime_controller_respects_budget_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dim = 13
            model = ActionRiskPolicyNet(dim, hidden_dim=8)
            checkpoint = {
                "model": model.state_dict(),
                "dim": dim,
                "hidden_dim": 8,
                "ratios": DEFAULT_RATIOS,
                "metadata": {
                    "feature_schema": POLICY_FEATURE_SCHEMA,
                    "calibrated_threshold": 0.5,
                },
            }
            torch.save(checkpoint, root / "policy.pt")
            controller = RaspTrainPolicyController(
                checkpoint_path=str(root / "policy.pt"),
                dataset="gsm8k",
                target_average_ratio=0.1,
                max_new_tokens=128,
                default_max_ratio=0.4,
                early_tokens=64,
                early_max_ratio=0.05,
            )
            ratio = controller.choose_ratio(
                RuntimeObservation(
                    generated_tokens=0,
                    entropy=0.1,
                    confidence=0.9,
                    hidden_state=torch.zeros(1, 4),
                )
            )
            self.assertLessEqual(ratio, 0.05)
            self.assertLessEqual(ratio, 0.1)


if __name__ == "__main__":
    unittest.main()
