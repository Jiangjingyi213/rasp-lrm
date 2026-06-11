from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.metrics.answer_match import answer_match
from src.utils.io import read_jsonl, write_json


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset") or "unknown"), str(row["id"]))


def _correct(row: dict[str, Any]) -> bool:
    return answer_match(str(row.get("completion", "")), str(row.get("gold", "")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dense_rows = read_jsonl(args.dense)
    policy_rows = read_jsonl(args.policy)
    dense_by_key = {_key(row): row for row in dense_rows}
    policy_by_key = {_key(row): row for row in policy_rows}
    if set(dense_by_key) != set(policy_by_key):
        missing_policy = sorted(set(dense_by_key) - set(policy_by_key))
        missing_dense = sorted(set(policy_by_key) - set(dense_by_key))
        raise ValueError(
            f"Dense/policy examples differ: missing_policy={missing_policy[:5]}, "
            f"missing_dense={missing_dense[:5]}"
        )

    paired = [(dense_by_key[key], policy_by_key[key]) for key in sorted(dense_by_key)]
    dense_correct = sum(int(_correct(dense)) for dense, _policy in paired)
    policy_correct = sum(int(_correct(policy)) for _dense, policy in paired)
    dense_correct_policy_wrong = [
        {
            "dataset": dense.get("dataset"),
            "id": dense["id"],
            "question": dense.get("question"),
            "dense_prediction": dense.get("prediction"),
            "policy_prediction": policy.get("prediction"),
            "policy_average_ratio": policy.get("runtime", {})
            .get("runtime_mlp", {})
            .get("average_decode_pruning_ratio"),
        }
        for dense, policy in paired
        if _correct(dense) and not _correct(policy)
    ]
    dense_wrong_policy_correct = sum(
        int(not _correct(dense) and _correct(policy))
        for dense, policy in paired
    )
    output = {
        "examples": len(paired),
        "dense_correct": dense_correct,
        "dense_accuracy": dense_correct / len(paired) if paired else None,
        "policy_correct": policy_correct,
        "policy_accuracy": policy_correct / len(paired) if paired else None,
        "correctness_recomputed_from_completion": True,
        "dense_correct_policy_wrong_count": len(dense_correct_policy_wrong),
        "dense_wrong_policy_correct_count": dense_wrong_policy_correct,
        "dense_correct_policy_wrong": dense_correct_policy_wrong,
        "dense_path": str(Path(args.dense)),
        "policy_path": str(Path(args.policy)),
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
