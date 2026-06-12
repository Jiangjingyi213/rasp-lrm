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
                        "action_window_alignment": "affected_next_token_decisions_v2",
                        "ranking_scope": "initial_prompt_prefill_fixed",
                        "generated_tokens_at_boundary": 0,
                        "position": 0.0,
                        "max_new_tokens": 512,
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
                    "aligned_window_bank": {
                        "ratios": [0.0, 0.1],
                        "window_tokens": 16,
                        "max_boundaries_per_example": 12,
                    },
                    "paths": {
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["boundaries"], 1)
        self.assertEqual(summary["configured_max_boundaries_per_example"], 12)
        self.assertEqual(summary["action_window_alignment"], "affected_next_token_decisions_v2")

    def test_stage_sensitivity_metadata_is_validated(self) -> None:
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
                        "action_window_alignment": "affected_next_token_decisions_v2",
                        "ranking_scope": "initial_prompt_prefill_fixed",
                        "generated_tokens_at_boundary": 0,
                        "position": 0.0,
                        "max_new_tokens": 512,
                        "window_token_divergence": 0.0,
                        "window_end_hidden_l2": 0.0,
                        "boundary_token_source": "trajectory_generated_token_ids",
                        "dense_control_flipped_from_baseline": False,
                        "operational_stage": "reasoning",
                        "stage_source": "hidden_stage_probe",
                        "reasoning_accepted": True,
                        "stage_probabilities": {"setup": 0.1, "reasoning": 0.8, "final": 0.1},
                    }
                )
            write_jsonl(root / "counterfactuals.jsonl", rows)
            write_jsonl(root / "probe.jsonl", [{**row, "hidden_index": i} for i, row in enumerate(rows)])
            (root / "hidden.pt").touch()
            summary = validate_aligned_window_bank(
                {
                    "aligned_window_bank": {"ratios": [0.0, 0.1]},
                    "stage_sensitivity": {"checkpoint": "fake", "reasoning_threshold": 0.7},
                    "paths": {
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["operational_stage_counts"], {"reasoning": 1})


if __name__ == "__main__":
    unittest.main()
