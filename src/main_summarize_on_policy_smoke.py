from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sources: dict[str, dict[str, Any]] = {}
    ratio_rows = Counter()
    ratio_flips = Counter()
    ratio_terminal_eos = Counter()
    validations_ok = True
    behavior_tags = set()
    for item in manifest:
        dataset = str(item["dataset"])
        run_dir = Path(item["run_dir"])
        validation = read_json(run_dir / "03_on_policy_validation.json")
        rows = read_jsonl(run_dir / "01_on_policy_dataset.jsonl")
        behavior_tags.add(str(validation["behavior_policy_tag"]))
        validations_ok = validations_ok and validation.get("status") == "ok"
        sources[dataset] = {
            "status": validation.get("status"),
            "valid_problems": int(validation.get("valid_problems", 0)),
            "boundaries": int(validation.get("boundaries", 0)),
            "replay_failures": int(validation.get("replay_failures", 0)),
            "invalid_candidate_boundaries": int(
                validation.get("invalid_candidate_boundaries", 0)
            ),
            "rows_with_prior_action": int(validation.get("rows_with_prior_action", 0)),
            "rows_with_correct_on_policy_dense_control": int(
                validation.get("rows_with_correct_on_policy_dense_control", 0)
            ),
            "replay_components_verified": validation.get(
                "replay_components_verified", []
            ),
            "checks": validation.get("checks", {}),
        }
        for row in rows:
            for ratio, flipped, terminal in zip(
                row["candidate_ratios"][1:],
                row["candidate_flipped_from_on_policy_dense_control"][1:],
                row["candidate_terminal_eos"][1:],
            ):
                name = f"{float(ratio):.2f}"
                ratio_rows[name] += 1
                ratio_flips[name] += int(bool(flipped))
                ratio_terminal_eos[name] += int(bool(terminal))

    checks = {
        "both_training_sources_present": set(sources) == {"gsm8k", "math_train"},
        "single_behavior_policy": len(behavior_tags) == 1,
        "all_source_validations_passed": validations_ok,
        "minimum_four_valid_problems_per_source": all(
            int(value["valid_problems"]) >= 4 for value in sources.values()
        ),
        "all_rows_are_prior_action_states": all(
            int(value["boundaries"]) == int(value["rows_with_prior_action"])
            for value in sources.values()
        ),
        "all_replays_exact": all(int(value["replay_failures"]) == 0 for value in sources.values()),
    }
    passed = all(checks.values())
    write_json(
        root / "on_policy_smoke_summary.json",
        {
            "schema": "rasp_on_policy_action_risk_smoke_aggregate_v1",
            "passed": passed,
            "behavior_policy_tag": next(iter(behavior_tags)) if len(behavior_tags) == 1 else None,
            "sources": sources,
            "candidate_dose_response": [
                {
                    "ratio": float(name),
                    "rows": int(ratio_rows[name]),
                    "flips": int(ratio_flips[name]),
                    "flip_rate": ratio_flips[name] / max(1, ratio_rows[name]),
                    "terminal_eos": int(ratio_terminal_eos[name]),
                    "terminal_eos_rate": ratio_terminal_eos[name]
                    / max(1, ratio_rows[name]),
                }
                for name in sorted(ratio_rows)
            ],
            "checks": checks,
            "on_policy_bank_expansion_allowed": passed,
            "learned_multi_window_allowed": False,
            "final_test_sources_used": False,
        },
    )
    write_json(
        root / "phase_gate.json",
        {
            "schema": "rasp_on_policy_action_risk_smoke_gate_v1",
            "passed": passed,
            "checks": checks,
            "next_step": (
                "Expand the on-policy bank and run grouped OOF before learned multi-window."
                if passed
                else "Stop and inspect on-policy replay integrity."
            ),
        },
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
