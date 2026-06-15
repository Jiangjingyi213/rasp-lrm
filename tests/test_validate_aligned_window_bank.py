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
                        "stage_position": 0.0,
                        "stage_position_definition": "generated_tokens_over_dense_trajectory_tokens_minus_one",
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

    def test_empty_shard_passes_only_when_source_has_no_expected_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "counterfactuals.jsonl", [])
            write_jsonl(root / "probe.jsonl", [])
            write_jsonl(
                root / "trajectories.jsonl",
                [
                    {
                        "dataset": "fake",
                        "id": "p0",
                        "completion": "1",
                        "gold": "2",
                        "generated_token_ids": list(range(64)),
                    }
                ],
            )
            (root / "hidden.pt").touch()
            summary = validate_aligned_window_bank(
                {
                    "generation": {"max_new_tokens": 512},
                    "aligned_window_bank": {
                        "ratios": [0.0, 0.1],
                        "window_tokens": 16,
                        "boundary_sampling": "causal_grid",
                        "decision_start": 32,
                        "decision_stride": 32,
                    },
                    "paths": {
                        "trajectories": str(root / "trajectories.jsonl"),
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(summary["empty_shard"])
        self.assertTrue(summary["empty_shard_verified"])

    def test_empty_shard_fails_when_dense_correct_trajectory_has_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "counterfactuals.jsonl", [])
            write_jsonl(root / "probe.jsonl", [])
            write_jsonl(
                root / "trajectories.jsonl",
                [
                    {
                        "dataset": "fake",
                        "id": "p0",
                        "completion": "42",
                        "gold": "42",
                        "generated_token_ids": list(range(64)),
                    }
                ],
            )
            (root / "hidden.pt").touch()
            summary = validate_aligned_window_bank(
                {
                    "generation": {"max_new_tokens": 512},
                    "aligned_window_bank": {
                        "ratios": [0.0, 0.1],
                        "window_tokens": 16,
                        "boundary_sampling": "causal_grid",
                        "decision_start": 32,
                        "decision_stride": 32,
                    },
                    "paths": {
                        "trajectories": str(root / "trajectories.jsonl"),
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["empty_shard_verified"])
        self.assertIn(
            "Aligned bank boundary coverage does not match configured dense-correct trajectories",
            summary["errors"],
        )

    def test_causal_grid_rejects_completed_window_marked_as_terminal_eos(self) -> None:
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
                        "generated_tokens_at_boundary": 32,
                        "position": 32 / 512,
                        "max_new_tokens": 512,
                        "window_token_divergence": 0.0,
                        "window_end_hidden_l2": 0.0,
                        "boundary_token_source": "trajectory_generated_token_ids",
                        "dense_control_flipped_from_baseline": False,
                        "controller_eligible": True,
                        "action_duration_tokens": 16,
                        "action_completed_or_terminal": True,
                        "dense_restored_after_window": True,
                        "action_terminal_eos": True,
                    }
                )
            write_jsonl(root / "counterfactuals.jsonl", rows)
            write_jsonl(
                root / "probe.jsonl",
                [{**row, "hidden_index": index} for index, row in enumerate(rows)],
            )
            write_jsonl(
                root / "trajectories.jsonl",
                [
                    {
                        "dataset": "fake",
                        "id": "p0",
                        "completion": "42",
                        "gold": "42",
                        "generated_token_ids": list(range(64)),
                    }
                ],
            )
            (root / "hidden.pt").touch()
            summary = validate_aligned_window_bank(
                {
                    "generation": {"max_new_tokens": 512},
                    "aligned_window_bank": {
                        "ratios": [0.0, 0.1],
                        "window_tokens": 16,
                        "boundary_sampling": "causal_grid",
                        "decision_start": 32,
                        "decision_stride": 32,
                    },
                    "paths": {
                        "trajectories": str(root / "trajectories.jsonl"),
                        "counterfactuals": str(root / "counterfactuals.jsonl"),
                        "probe_dataset": str(root / "probe.jsonl"),
                        "probe_hidden_states": str(root / "hidden.pt"),
                    },
                }
            )
        self.assertEqual(summary["status"], "failed")
        self.assertIn(
            "A completed action window cannot also terminate early by EOS",
            summary["errors"],
        )


if __name__ == "__main__":
    unittest.main()
