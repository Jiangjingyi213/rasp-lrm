from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.stage_calibration.final_shards import (
    merge_final_shards,
    shard_dataset_dir,
    shard_summary_path,
    shard_tasks,
    summarize_rows,
)
from src.utils.io import read_json, write_json, write_jsonl


def row(index: int, *, correct: bool, method: str = "candidate") -> dict:
    return {
        "id": f"task-{index}",
        "final_eval_index": index,
        "correct": correct,
        "stage_protocol": {"valid": True},
        "runtime_stage_mask": {
            "tokens_by_stage": {"setup": 1, "reasoning": 2, "verify": 1, "final": 1},
            "fallback_reason": None,
            "theoretical_average_mlp_pruning_ratio": 0.2 if method != "structured_dense" else 0.0,
        },
        "truncated": False,
        "generated_tokens": 5,
    }


def method(name: str) -> dict:
    return {
        "name": name,
        "policy": "stage_specific",
        "stage_ratios": {"setup": 0.2, "reasoning": 0.2, "verify": 0.2, "final": 0.2},
        "prompt": {},
        "bias_compensation": True,
    }


class StageFinalShardsTest(unittest.TestCase):
    def test_shard_tasks_covers_each_item_once(self) -> None:
        tasks = [{"id": index} for index in range(10)]
        shards = [shard_tasks(tasks, shard_index=index, shard_count=4) for index in range(4)]
        covered = sorted(item["final_eval_index"] for shard in shards for item in shard)
        self.assertEqual(covered, list(range(10)))
        self.assertEqual([item["final_eval_index"] for item in shards[0]], [0, 4, 8])

    def test_summarize_rows(self) -> None:
        summary = summarize_rows(
            [row(0, correct=True), row(1, correct=False)],
            method=method("candidate"),
            seed=7,
        )
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["problems"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["fallback_rate"], 0.0)
        self.assertAlmostEqual(summary["theoretical_average_mlp_pruning_ratio"], 0.2)

    def test_merge_final_shards_writes_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            final_dir = Path(tmpdir) / "06_final"
            shard_count = 2
            for shard_index in range(shard_count):
                write_json(
                    shard_summary_path(final_dir, shard_index=shard_index, shard_count=shard_count),
                    {"schema": "stage_calibrated_final_eval_v1", "final_shard": {"index": shard_index, "count": shard_count}},
                )
                shard_dir = shard_dataset_dir(
                    final_dir,
                    "gsm8k",
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
                dense_rows = [
                    row(index, correct=(index != 3), method="structured_dense")
                    for index in range(shard_index, 4, shard_count)
                ]
                candidate_rows = [
                    row(index, correct=(index in {0, 2}), method="candidate")
                    for index in range(shard_index, 4, shard_count)
                ]
                write_jsonl(shard_dir / "structured_dense_seed1.jsonl", dense_rows)
                write_json(
                    shard_dir / "structured_dense_seed1.summary.json",
                    summarize_rows(dense_rows, method=method("structured_dense"), seed=1),
                )
                write_jsonl(shard_dir / "candidate_seed1.jsonl", candidate_rows)
                write_json(
                    shard_dir / "candidate_seed1.summary.json",
                    summarize_rows(candidate_rows, method=method("candidate"), seed=1),
                )

            output = final_dir / "summary.json"
            merge_final_shards(
                final_dir=final_dir,
                shard_count=shard_count,
                output_summary_path=output,
                metadata={"config_hash": "abc", "model_name": "model", "profile": "pilot"},
                final_eval_limit=4,
                bootstrap_seed=1,
            )
            summary = read_json(output)
            self.assertTrue(summary["final_sharded"])
            self.assertEqual(summary["aggregates"]["gsm8k"]["candidate"]["accuracy_mean"], 0.5)
            self.assertEqual(
                summary["aggregates"]["gsm8k"]["candidate"]["paired_accuracy_delta_vs_structured_dense"],
                -0.25,
            )
            combined = Path(tmpdir) / "06_final" / "gsm8k" / "candidate_seed1.jsonl"
            self.assertTrue(combined.exists())


if __name__ == "__main__":
    unittest.main()
