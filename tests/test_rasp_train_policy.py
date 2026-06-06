from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.probes.rasp_train_dataset import DEFAULT_RATIOS, RaspTrainPolicyDataset, ratio_index
from src.main_train_rasp_train_router import policy_split_summary
from src.rasp.budget_controller import RuntimeObservation
from src.rasp.train_controller import RaspTrainPolicyController
from src.rasp.train_policy import POLICY_FEATURE_SCHEMA, RatioPolicyNet, ratio_policy_loss
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
            features, label, unsafe_mask, ratios, target_budget, index = dataset[0]
            self.assertEqual(index, 0)
            self.assertEqual(int(label), DEFAULT_RATIOS.index(0.1))
            self.assertAlmostEqual(float(target_budget), 0.2)
            self.assertEqual(tuple(unsafe_mask.shape), (len(DEFAULT_RATIOS),))
            self.assertEqual(tuple(ratios.shape), (len(DEFAULT_RATIOS),))
            self.assertGreater(features.numel(), 4)

    def test_policy_split_summary_does_not_require_row_flip_label(self) -> None:
        rows = [
            {
                "dataset": "gsm8k",
                "id": "train",
                "candidate_unsafe": [False, True],
            },
            {
                "dataset": "gsm8k",
                "id": "val",
                "candidate_unsafe": [False, False],
            },
        ]
        summary = policy_split_summary(rows, [0], [1])
        self.assertEqual(summary["train_problem_count"], 1)
        self.assertEqual(summary["val_problem_count"], 1)
        self.assertEqual(summary["unsafe_candidate_rate_train"], 0.5)

    def test_loss_accepts_unsafe_and_budget_terms(self) -> None:
        logits = torch.zeros(2, len(DEFAULT_RATIOS))
        labels = torch.tensor([0, 3])
        unsafe_mask = torch.zeros(2, len(DEFAULT_RATIOS))
        unsafe_mask[:, -1] = 1.0
        ratios = torch.tensor([DEFAULT_RATIOS, DEFAULT_RATIOS], dtype=torch.float32)
        target_budget = torch.tensor([0.2, 0.2])
        loss = ratio_policy_loss(
            logits,
            labels,
            unsafe_mask,
            ratios,
            target_budget,
            unsafe_weight=2.0,
            budget_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_runtime_controller_respects_budget_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dim = 13
            model = RatioPolicyNet(dim, len(DEFAULT_RATIOS), hidden_dim=8)
            checkpoint = {
                "model": model.state_dict(),
                "dim": dim,
                "num_ratios": len(DEFAULT_RATIOS),
                "hidden_dim": 8,
                "ratios": DEFAULT_RATIOS,
                "metadata": {
                    "feature_schema": POLICY_FEATURE_SCHEMA,
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
