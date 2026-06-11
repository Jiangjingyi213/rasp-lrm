from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("dataset", "id", "segment_id")


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[field].strip() for field in KEY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument(
        "--labels",
        default="configs/stage_audits/s1_three_stage_v3_labels.csv",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit)
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(
            f"Reviewed audit labels not found: {labels_path}. "
            "Complete and sync the independent audit before training."
        )
    with audit_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = {key(row): row for row in csv.DictReader(handle)}
    audit_keys = {key(row) for row in rows}
    if audit_keys != set(labels):
        raise SystemExit(
            f"Audit/label keys differ: missing_labels={len(audit_keys - set(labels))}, "
            f"unknown_labels={len(set(labels) - audit_keys)}"
        )
    for row in rows:
        label = labels[key(row)]
        if row["stage"].strip() != label["stage"].strip():
            raise SystemExit(f"Rule stage changed for {key(row)}; regenerate and re-audit")
        row["audited_stage"] = label["audited_stage"].strip()
        row["audit_status"] = label["audit_status"].strip()
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Applied {len(rows)} reviewed stage labels to {audit_path}")


if __name__ == "__main__":
    main()
