from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from src.stage_calibration.mask_bank import build_mask_bank, load_mask_bank, save_mask_bank, validate_mask_bank
from src.stage_calibration.protocol import STAGES
from src.stage_calibration.statistics import TokenMoments, keep_mask, wifv


class StageCalibrationStatisticsTest(unittest.TestCase):
    def test_token_moments_and_wifv(self) -> None:
        moment = TokenMoments.zeros(2)
        moment.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.assertTrue(torch.allclose(moment.mean(), torch.tensor([2.0, 3.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(moment.variance(), torch.tensor([1.0, 1.0], dtype=torch.float64)))
        values = wifv(moment, torch.tensor([[2.0, 3.0]]))
        self.assertTrue(torch.allclose(values, torch.tensor([4.0, 9.0])))

    def test_keep_masks_are_nested(self) -> None:
        values = torch.arange(20, dtype=torch.float32)
        small = keep_mask(values, 0.10)
        large = keep_mask(values, 0.20)
        self.assertFalse(bool((large & ~small).any()))

    def test_mask_bank_validates_ratio_and_stage_layout(self) -> None:
        sources = ("c4", "prompt_only", "trajectory", *STAGES)
        metrics = {source: {0: torch.arange(20, dtype=torch.float32)} for source in sources}
        means = {source: {0: torch.zeros(20)} for source in sources}
        bank = build_mask_bank(metadata={}, metrics=metrics, means=means, ratios=[0.0, 0.1, 0.2])
        validate_mask_bank(bank)
        self.assertIn("stage_specific_al_am", bank["policies"])

    def test_mask_bank_rejects_metadata_mismatch(self) -> None:
        sources = ("c4", "prompt_only", "trajectory", *STAGES)
        metrics = {source: {0: torch.arange(20, dtype=torch.float32)} for source in sources}
        means = {source: {0: torch.zeros(20)} for source in sources}
        bank = build_mask_bank(metadata={"model_name": "a"}, metrics=metrics, means=means, ratios=[0.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.pt"
            save_mask_bank(path, bank)
            with self.assertRaises(ValueError):
                load_mask_bank(path, {"model_name": "b"})


if __name__ == "__main__":
    unittest.main()
