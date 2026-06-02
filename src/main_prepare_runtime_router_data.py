from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from src.rasp.safe_oracle import allocate_budget_aware_safe_oracle, build_safe_oracle_steps
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def _load_shards(run_dirs: list[str]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows = []
    hidden_parts = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        validation_path = root / "07_runtime_bank_validation.json"
        if not validation_path.exists():
            raise ValueError(f"{root}: missing runtime-bank validation summary")
        validation = read_json(validation_path)
        if validation.get("status") != "ok":
            raise ValueError(f"{root}: runtime-bank validation status is not ok")
        shard_rows = read_jsonl(root / "05_probe_dataset.jsonl")
        shard_hidden = torch.load(root / "05_probe_hidden_states.pt", map_location="cpu")
        if len(shard_rows) != len(shard_hidden):
            raise ValueError(f"{root}: probe rows ({len(shard_rows)}) and hidden states ({len(shard_hidden)}) differ")
        for row in shard_rows:
            out = dict(row)
            out["source_run_dir"] = str(root)
            rows.append(out)
        hidden_parts.append(shard_hidden)
    if not rows:
        raise ValueError("No runtime-bank rows found")
    return rows, torch.cat(hidden_parts, dim=0)


def _oracle_summary(selected: list[dict[str, Any]], target: float, ratio_field: str) -> dict[str, Any]:
    ratios = [float(row["selected_ratio"]) for row in selected]
    return {
        "target_average_ratio": target,
        "ratio_field": ratio_field,
        "n_problem_steps": len(selected),
        "average_selected_ratio": sum(ratios) / len(ratios) if ratios else None,
        "budget_utilization": sum(ratios) / (len(ratios) * target) if ratios and target else None,
        "ratio_distribution": dict(sorted(Counter(f"{ratio:.2f}" for ratio in ratios).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows, hidden = _load_shards(args.run_dirs)
    for index, row in enumerate(rows):
        row["hidden_index"] = index
    safe_steps = build_safe_oracle_steps(rows)
    risk_rows = [dict(row) for row in rows if float(row.get("ratio", 0.0)) > 0.0]
    risk_hidden = torch.stack([hidden[int(row["hidden_index"])] for row in risk_rows])
    for index, row in enumerate(risk_rows):
        row["hidden_index"] = index

    write_jsonl(output_dir / "05_probe_dataset_merged.jsonl", rows)
    torch.save(hidden, output_dir / "05_probe_hidden_states_merged.pt")
    write_jsonl(output_dir / "09_action_conditioned_risk_dataset.jsonl", risk_rows)
    torch.save(risk_hidden, output_dir / "09_action_conditioned_risk_hidden_states.pt")
    write_jsonl(output_dir / "08_safe_oracle_steps.jsonl", safe_steps)

    budget_summaries = []
    for ratio_field in ("max_safe_ratio", "monotonic_safe_ratio"):
        for budget in args.budgets:
            selected = allocate_budget_aware_safe_oracle(safe_steps, budget, ratio_field=ratio_field)
            suffix = f"{ratio_field}_b{budget:.2f}"
            write_jsonl(output_dir / f"08_budget_aware_safe_oracle_{suffix}.jsonl", selected)
            budget_summaries.append(_oracle_summary(selected, budget, ratio_field))

    summary = {
        "method": "runtime_rasp_zero_router_data",
        "source_run_dirs": args.run_dirs,
        "counterfactual_rows_with_dense_control": len(rows),
        "action_conditioned_risk_rows": len(risk_rows),
        "problem_count": len({(str(row.get("dataset")), str(row["id"])) for row in rows}),
        "problem_step_count": len(safe_steps),
        "positive_rate": sum(int(bool(row["flipped"])) for row in risk_rows) / len(risk_rows),
        "non_monotonic_step_count": sum(int(bool(row["non_monotonic"])) for row in safe_steps),
        "budget_aware_safe_oracles": budget_summaries,
    }
    write_json(output_dir / "00_runtime_router_data_summary.json", summary)


if __name__ == "__main__":
    main()
