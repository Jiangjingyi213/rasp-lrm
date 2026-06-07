from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.main_validate_aligned_window_bank import validate_aligned_window_bank
from src.utils.io import write_jsonl


class ValidateAlignedWindowBankTest(unittest.TestCase):
    def test_valid_aligned_bank_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for ratio in (0.0, 0.1):
                rows.append(
                    {
                        "dataset": "fake",
                        "id": "p0",
                        "boundary_index": 0,
                        "ratio": ratio,
                        "flipped": False,
                        "action_scope": "single_fixed_window_then_dense",
                        "ranking_scope": "initial_prompt_prefill_fixed",
                        "window_token_divergence": 0.0,
                        "window_end_hidden_l2": 0.0,
                        "boundary_token_source": "trajectory_generated_token_ids",
                        "dense_control_flipped_from_baseline": False,
                    }
                )
            write_jsonl(root / "counterfactuals.jsonl", rows)
            write_jsonl(
                root / "probe.jsonl",
                [{**row, "hidden_index": index} for index, row in enumerate(rows)],
            )
            (root / "hidden.pt").touch()
            summary = validate_aligned_window_bank(
                {
                    "aligned_window_bank": {"ratios": [0.0, 0.1]},
                    "paths": {
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["boundaries"], 1)


if __name__ == "__main__":
    unittest.main()
