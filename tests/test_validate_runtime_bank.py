from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.main_validate_runtime_bank import validate_runtime_bank
from src.utils.io import write_jsonl


class ValidateRuntimeBankTest(unittest.TestCase):
    def _config(self, root: Path, ratio_zero_flipped: bool = False) -> dict:
        paths = {
            "trajectories": str(root / "01_trajectories.jsonl"),
            "segments": str(root / "02_segments.jsonl"),
            "counterfactuals": str(root / "03_counterfactuals.jsonl"),
            "probe_dataset": str(root / "05_probe_dataset.jsonl"),
            "probe_hidden_states": str(root / "05_probe_hidden_states.pt"),
        }
        write_jsonl(
            paths["trajectories"],
            [{"id": "p0", "correct": True, "completion": "Step 1: calculate.\n\nFinal answer: 3"}],
        )
        write_jsonl(
            paths["segments"],
            [
                {
                    "id": "p0",
                    "completion": "Step 1: calculate.\n\nFinal answer: 3",
                    "segments": [
                        {"segment_id": 0, "segment_type": "derivation", "text": "Step 1: calculate."},
                        {"segment_id": 1, "segment_type": "final", "text": "Final answer: 3"},
                    ],
                }
            ],
        )
        counterfactuals = []
        for segment_id in (0, 1):
            for ratio in (0.0, 0.1):
                counterfactuals.append(
                    {
                        "dataset": "fake",
                        "id": "p0",
                        "segment_id": segment_id,
                        "module": "mlp_intermediate_channels",
                        "ratio": ratio,
                        "pruned_layers": [0, 1],
                        "flipped": bool(ratio == 0.0 and ratio_zero_flipped),
                    }
                )
        write_jsonl(paths["counterfactuals"], counterfactuals)
        write_jsonl(
            paths["probe_dataset"],
            [{**row, "hidden_index": index} for index, row in enumerate(counterfactuals)],
        )
        Path(paths["probe_hidden_states"]).touch()
        return {
            "paths": paths,
            "counterfactual": {"ratios": [0.0, 0.1], "layers": [0, 1]},
            "runtime_bank_validation": {
                "module": "mlp_intermediate_channels",
                "require_all_dense_correct": True,
                "max_ratio_zero_flip_rate": 0.05,
            },
        }

    def test_complete_runtime_bank_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = validate_runtime_bank(self._config(Path(tmp)))
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["reasoning_steps"], 2)
        self.assertEqual(summary["ratio_zero_flip_rate"], 0.0)

    def test_ratio_zero_flip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = validate_runtime_bank(self._config(Path(tmp), ratio_zero_flipped=True))
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("ratio=0" in error for error in summary["errors"]))

    def test_ratio_zero_flip_can_be_allowed_for_later_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), ratio_zero_flipped=True)
            cfg["runtime_bank_validation"]["allow_ratio_zero_filtering"] = True
            summary = validate_runtime_bank(cfg)
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(any("ratio=0" in warning for warning in summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
