from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.metrics.answer_match import answer_match, extract_answer, math_verify_available
from src.utils.io import append_jsonl, ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    if not math_verify_available():
        raise RuntimeError(
            "Math-Verify is not installed. Install requirements.txt before rescoring."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.unlink(missing_ok=True)

    total = 0
    old_correct = 0
    new_correct = 0
    changed_to_correct = 0
    changed_to_wrong = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            previous = bool(row.get("correct", False))
            current = answer_match(str(row.get("completion", "")), str(row.get("gold", "")))
            row["prediction"] = extract_answer(str(row.get("completion", "")))
            row["correct"] = current
            row["grader"] = "lightweight_plus_math_verify"
            append_jsonl(output_path, row)
            total += 1
            old_correct += int(previous)
            new_correct += int(current)
            changed_to_correct += int(not previous and current)
            changed_to_wrong += int(previous and not current)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "grader": "lightweight_plus_math_verify",
        "total": total,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy": old_correct / total if total else None,
        "new_accuracy": new_correct / total if total else None,
        "changed_to_correct": changed_to_correct,
        "changed_to_wrong": changed_to_wrong,
    }
    summary_path = args.summary or str(output_path.with_suffix(".summary.json"))
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
