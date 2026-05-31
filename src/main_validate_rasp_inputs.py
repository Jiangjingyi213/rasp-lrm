from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODULES = ["layer", "attention_block", "mlp_block", "attention_heads", "mlp_channels"]
DEFAULT_RATIOS = [0.2, 0.4, 0.6]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL row in {path}:{line_number}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_run(run_dir: str, modules: list[str], ratios: list[float]) -> dict[str, Any]:
    root = Path(run_dir)
    probe_rows = read_jsonl(root / "05_probe_dataset.jsonl")
    oracle = read_json(root / "03_counterfactuals.oracles.json")
    expected_rows = int(oracle["n_counterfactuals"])
    if len(probe_rows) != expected_rows:
        raise ValueError(
            f"Incomplete probe table in {root}: expected {expected_rows} rows "
            "from 03_counterfactuals.oracles.json, "
            f"found {len(probe_rows)}"
        )
    expected_actions = {(module, float(ratio)) for module in modules for ratio in ratios}
    actions_by_step: dict[tuple[str, str, int], set[tuple[str, float]]] = {}
    for row in probe_rows:
        key = (str(row.get("dataset") or "unknown"), str(row["id"]), int(row["segment_id"]))
        actions_by_step.setdefault(key, set()).add((str(row["module"]), float(row["ratio"])))
    incomplete_steps = [key for key, actions in actions_by_step.items() if actions != expected_actions]
    if incomplete_steps:
        raise ValueError(f"Incomplete action grid in {root}: {len(incomplete_steps)} problem-steps are missing actions")
    required_files = [
        root / "05_probe_hidden_states.pt",
        root / "05_probe_activation_features.pt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise ValueError(f"Missing probe tensors in {root}: {missing}")
    return {
        "run_dir": str(root),
        "problem_count": len({(str(row.get("dataset") or "unknown"), str(row["id"])) for row in probe_rows}),
        "segment_count": len(actions_by_step),
        "probe_rows": len(probe_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()
    summaries = [validate_run(run_dir, modules=args.modules, ratios=args.ratios) for run_dir in args.run_dirs]
    for summary in summaries:
        print(
            f"OK {summary['run_dir']}: "
            f"{summary['problem_count']} problems, {summary['segment_count']} segments, {summary['probe_rows']} probe rows"
        )
    print("RASP input validation passed.")


if __name__ == "__main__":
    main()
