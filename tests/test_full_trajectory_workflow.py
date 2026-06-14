from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main_summarize_full_trajectory_workflow import main
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, write_json


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


if __name__ == "__main__":
    unittest.main()
