from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main_summarize_full_trajectory_workflow import main
from src.main_summarize_on_policy_smoke import main as summarize_on_policy
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, write_json, write_jsonl


class FullTrajectoryWorkflowTest(unittest.TestCase):
    def test_config_fingerprint_changes_with_runtime_contract(self) -> None:
        left = {"runtime_rasp": {"fixed_ratio": 0.1}, "data": {"limit": 20}}
        right = {"runtime_rasp": {"fixed_ratio": 0.2}, "data": {"limit": 20}}
        sections = ("data", "runtime_rasp")
        self.assertNotEqual(
            config_fingerprint(left, sections),
            config_fingerprint(right, sections),
        )

    def test_final_summary_never_enables_learned_multi_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "sys.argv",
                [
                    "main_summarize_full_trajectory_workflow",
                    "--root",
                    str(root),
                    "--failed-stage",
                    "dense_bank_smoke_collect",
                ],
            ):
                main()
            gate = read_json(root / "workflow_gate.json")
            summary = read_json(root / "final_workflow_summary.json")
            self.assertFalse(gate["learned_multi_window_allowed"])
            self.assertFalse(summary["learned_multi_window_allowed"])
            self.assertEqual(gate["failed_stage"], "dense_bank_smoke_collect")
            self.assertTrue((root / "final_workflow_report_zh.md").is_file())

    def test_completed_smoke_only_allows_on_policy_bank_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase in (
                "00_preflight",
                "01_dense_bank_smoke",
                "02_dense_bank_pilot",
                "03_fixed_multi_window_dev",
                "04_on_policy_smoke",
            ):
                write_json(root / phase / "phase_gate.json", {"passed": True, "checks": {}})
            write_json(
                root / "00_preflight" / "00_existing_bank_preflight.json",
                {"preflight_valid": True},
            )
            write_json(
                root
                / "01_dense_bank_smoke"
                / "data"
                / "01_full_trajectory_data_summary.json",
                {},
            )
            write_json(
                root
                / "02_dense_bank_pilot"
                / "data"
                / "01_full_trajectory_data_summary.json",
                {},
            )
            write_json(
                root
                / "02_dense_bank_pilot"
                / "analysis"
                / "02_full_trajectory_analysis.json",
                {},
            )
            write_json(
                root / "03_fixed_multi_window_dev" / "fixed_multi_window_summary.json",
                {},
            )
            write_json(
                root / "03_fixed_multi_window_dev" / "selected_behavior_policy.json",
                {},
            )
            write_json(
                root / "04_on_policy_smoke" / "on_policy_smoke_summary.json",
                {"on_policy_bank_expansion_allowed": True},
            )
            with patch(
                "sys.argv",
                ["main_summarize_full_trajectory_workflow", "--root", str(root)],
            ):
                main()
            summary = read_json(root / "final_workflow_summary.json")
            self.assertTrue(summary["completed"])
            self.assertTrue(summary["on_policy_bank_expansion_allowed"])
            self.assertFalse(summary["learned_multi_window_allowed"])

    def test_on_policy_summary_separates_harmful_and_beneficial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = []
            for dataset in ("gsm8k", "math_train"):
                run_dir = root / dataset
                manifest.append({"dataset": dataset, "run_dir": str(run_dir)})
                write_json(
                    run_dir / "03_on_policy_validation.json",
                    {
                        "status": "ok",
                        "behavior_policy_tag": "fixed",
                        "risk_label_semantics": "harmful_flip_conditioned_on_correct_dense_control_v1",
                        "valid_problems": 4,
                        "boundaries": 2,
                        "replay_failures": 0,
                        "invalid_candidate_boundaries": 0,
                        "rows_with_prior_action": 2,
                        "rows_with_correct_on_policy_dense_control": 1,
                        "checks": {
                            "all_rows_originate_from_dense_correct_problems": True
                        },
                    },
                )
                write_jsonl(
                    run_dir / "01_on_policy_dataset.jsonl",
                    [
                        {
                            "candidate_ratios": [0.0, 0.1],
                            "candidate_flipped_from_on_policy_dense_control": [False, True],
                            "candidate_harmful_flip": [False, True],
                            "candidate_beneficial_correction": [False, False],
                            "candidate_terminal_eos": [False, False],
                            "on_policy_dense_control_correct": True,
                        },
                        {
                            "candidate_ratios": [0.0, 0.1],
                            "candidate_flipped_from_on_policy_dense_control": [False, True],
                            "candidate_harmful_flip": [False, False],
                            "candidate_beneficial_correction": [False, True],
                            "candidate_terminal_eos": [False, False],
                            "on_policy_dense_control_correct": False,
                        },
                    ],
                )
            manifest_path = root / "manifest.json"
            write_json(manifest_path, manifest)
            with patch(
                "sys.argv",
                [
                    "main_summarize_on_policy_smoke",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                ],
            ):
                summarize_on_policy()
            summary = read_json(root / "on_policy_smoke_summary.json")
            dose = summary["candidate_dose_response"][0]
            self.assertEqual(dose["answer_changes"], 4)
            self.assertEqual(dose["harmful_flips"], 2)
            self.assertEqual(dose["beneficial_corrections"], 2)
            self.assertEqual(dose["harmful_flip_rate_among_correct_controls"], 1.0)
            self.assertEqual(
                dose["beneficial_correction_rate_among_incorrect_controls"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
