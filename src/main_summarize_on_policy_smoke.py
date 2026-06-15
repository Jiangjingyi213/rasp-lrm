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
    ratio_answer_changes = Counter()
    ratio_harmful_flips = Counter()
    ratio_beneficial_corrections = Counter()
    ratio_correct_controls = Counter()
    ratio_incorrect_controls = Counter()
    ratio_terminal_eos = Counter()
    validations_ok = True
    behavior_tags = set()
    for item in manifest:
        dataset = str(item["dataset"])
        run_dir = Path(item["run_dir"])
        validation = read_json(run_dir / "03_on_policy_validation.json")
        rows = read_jsonl(run_dir / "01_on_policy_dataset.jsonl")
        behavior_tags.add(str(validation["behavior_policy_tag"]))
        validations_ok = (
            validations_ok
            and validation.get("status") == "ok"
            and validation.get("risk_label_semantics")
            == "harmful_flip_conditioned_on_correct_dense_control_v1"
        )
        sources[dataset] = {
            "status": validation.get("status"),
            "valid_problems": int(validation.get("valid_problems", 0)),
            "eligible_dense_correct_problems": int(
                validation.get("eligible_dense_correct_problems", 0)
            ),
            "eligible_behavior_correct_problems": int(
                validation.get("eligible_behavior_correct_problems", 0)
            ),
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
            for ratio, changed, harmful, beneficial, terminal in zip(
                row["candidate_ratios"][1:],
                row["candidate_flipped_from_on_policy_dense_control"][1:],
                row["candidate_harmful_flip"][1:],
                row["candidate_beneficial_correction"][1:],
                row["candidate_terminal_eos"][1:],
            ):
                name = f"{float(ratio):.2f}"
                ratio_rows[name] += 1
                ratio_answer_changes[name] += int(bool(changed))
                ratio_harmful_flips[name] += int(bool(harmful))
                ratio_beneficial_corrections[name] += int(bool(beneficial))
                ratio_correct_controls[name] += int(
                    bool(row["on_policy_dense_control_correct"])
                )
                ratio_incorrect_controls[name] += int(
                    not bool(row["on_policy_dense_control_correct"])
                )
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
        "all_rows_originate_from_dense_correct_problems": all(
            bool(value["checks"].get("all_rows_originate_from_dense_correct_problems"))
            for value in sources.values()
        ),
        "all_replays_exact": all(int(value["replay_failures"]) == 0 for value in sources.values()),
    }
    passed = all(checks.values())
    write_json(
        root / "on_policy_smoke_summary.json",
        {
            "schema": "rasp_on_policy_action_risk_smoke_aggregate_v1",
            "risk_label_semantics": "harmful_flip_conditioned_on_correct_dense_control_v1",
            "passed": passed,
            "behavior_policy_tag": next(iter(behavior_tags)) if len(behavior_tags) == 1 else None,
            "sources": sources,
            "candidate_dose_response": [
                {
                    "ratio": float(name),
                    "rows": int(ratio_rows[name]),
                    "answer_changes": int(ratio_answer_changes[name]),
                    "answer_change_rate": ratio_answer_changes[name]
                    / max(1, ratio_rows[name]),
                    "correct_dense_controls": int(ratio_correct_controls[name]),
                    "harmful_flips": int(ratio_harmful_flips[name]),
                    "harmful_flip_rate_among_correct_controls": ratio_harmful_flips[
                        name
                    ]
                    / max(1, ratio_correct_controls[name]),
                    "incorrect_dense_controls": int(ratio_incorrect_controls[name]),
                    "beneficial_corrections": int(ratio_beneficial_corrections[name]),
                    "beneficial_correction_rate_among_incorrect_controls": ratio_beneficial_corrections[
                        name
                    ]
                    / max(1, ratio_incorrect_controls[name]),
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
