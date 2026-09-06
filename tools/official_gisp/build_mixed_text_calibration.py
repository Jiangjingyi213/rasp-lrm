from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _text_from_row(row: dict[str, Any], fields: list[str]) -> str:
    pieces = []
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            pieces.append(value)
    text = "\n\n".join(pieces).strip()
    if not text:
        text = str(row.get("question", "")).strip()
    return text


def _stratified_half(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source") or row.get("dataset") or "unknown")].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for source in sorted(groups):
        group = list(groups[source])
        rng.shuffle(group)
        take = min(len(group), max(1, round(len(group) * samples / max(1, len(rows)))))
        selected.extend(group[:take])
        leftovers.extend(group[take:])

    if len(selected) > samples:
        rng.shuffle(selected)
        selected = selected[:samples]
    elif len(selected) < samples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: samples - len(selected)])

    selected.sort(key=lambda row: (str(row.get("source", "")), int(row.get("problem_index", 0)), int(row.get("sample_index", 0))))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a text-only half mixed calibration JSONL for official GISP local loader.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--text-fields", nargs="+", default=["prompt", "completion"])
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    samples = min(int(args.samples), len(rows))
    selected = _stratified_half(rows, samples, int(args.seed))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_counts: dict[str, int] = defaultdict(int)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            text = _text_from_row(row, list(args.text_fields))
            if not text:
                raise ValueError(f"Selected row has no usable text: {row.get('id', index)}")
            source = str(row.get("source") or row.get("dataset") or "unknown")
            source_counts[source] += 1
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "source": source,
                        "dataset": row.get("dataset"),
                        "source_dataset": row.get("source_dataset"),
                        "id": row.get("id"),
                        "original_index": index,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "schema": "official_gisp_mixed_text_calibration_v1",
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "samples": len(selected),
        "seed": int(args.seed),
        "text_fields": list(args.text_fields),
        "source_counts": dict(sorted(source_counts.items())),
        "contamination_policy": "Uses the existing mixed calibration pool; GSM8K/MATH/amc_aime/cn_k12 are excluded by the source config.",
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
