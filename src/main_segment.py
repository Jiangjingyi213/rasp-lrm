from __future__ import annotations

import argparse

from src.segmentation.rule_segmenter import segment_text
from src.utils.io import read_jsonl, read_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    in_path = args.input or cfg["paths"]["trajectories"]
    out_path = args.output or cfg["paths"]["segments"]
    min_chars = cfg.get("segmentation", {}).get("min_chars", 20)
    rows = []
    for item in read_jsonl(in_path):
        rows.append(
            {
                "id": item["id"],
                "dataset": item.get("dataset"),
                "question": item["question"],
                "gold": item.get("gold", ""),
                "completion": item["completion"],
                "prediction": item.get("prediction"),
                "correct": item.get("correct"),
                "segments": segment_text(item["completion"], min_chars=min_chars),
            }
        )
    write_jsonl(out_path, rows)


if __name__ == "__main__":
    main()
