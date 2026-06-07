from __future__ import annotations

import argparse
from pathlib import Path

from src.rasp.fair_benchmark import create_split_manifest, validate_equivalent_rows
from src.utils.io import ensure_dir, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b15-dataset", required=True)
    parser.add_argument("--b20-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    rows = read_jsonl(args.b15_dataset)
    other_rows = read_jsonl(args.b20_dataset)
    validate_equivalent_rows(rows, other_rows)
    output = ensure_dir(args.output_dir)
    for seed in args.seeds:
        manifest = create_split_manifest(rows, seed)
        manifest["source_b15_dataset"] = str(args.b15_dataset)
        manifest["source_b20_dataset"] = str(args.b20_dataset)
        write_json(Path(output) / f"seed_{seed}.json", manifest)
        print(f"wrote fair benchmark split seed={seed}: {manifest['problem_counts']}")


if __name__ == "__main__":
    main()
