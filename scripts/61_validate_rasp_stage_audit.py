from __future__ import annotations

import argparse
import csv
from pathlib import Path


STAGES = {"setup", "reasoning", "verification", "final"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--required-rows", type=int, default=100)
    parser.add_argument("--required-agreement", type=float, default=0.80)
    args = parser.parse_args()

    path = Path(args.audit)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if row.get("audited_stage", "").strip() in STAGES]
    agreement = (
        sum(row["stage"].strip() == row["audited_stage"].strip() for row in valid) / len(valid)
        if valid
        else 0.0
    )
    print(f"valid_audit_rows={len(valid)} agreement={agreement:.4f}")
    if len(valid) < args.required_rows or agreement < args.required_agreement:
        raise SystemExit(
            "Stage audit gate failed; do not train until the independent audit passes."
        )


if __name__ == "__main__":
    main()
