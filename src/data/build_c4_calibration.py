from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.utils.io import ensure_dir, write_json, write_jsonl


def _nonempty_text(row: dict[str, Any], text_field: str) -> str:
    text = str(row.get(text_field, "")).strip()
    return " ".join(text.split())


def _c4_direct_url(seed: int) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    shard_index = int(seed) % 1024
    return (
        f"{endpoint}/datasets/allenai/c4/resolve/main/en/"
        f"c4-train.{shard_index:05d}-of-01024.json.gz"
    )


def _iter_c4_direct_rows(seed: int) -> Any:
    url = os.environ.get("C4_DIRECT_URL") or _c4_direct_url(seed)
    request = Request(url, headers={"User-Agent": "rasp-lrm-c4-calibration"})
    with urlopen(request, timeout=int(os.environ.get("C4_DOWNLOAD_TIMEOUT", "300"))) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for line in gz:
                if not line.strip():
                    continue
                yield json.loads(line.decode("utf-8"))


def _iter_c4_dataset_rows(seed: int, buffer_size: int) -> Any:
    from datasets import load_dataset

    data_files = {"train": f"en/c4-train.{int(seed) % 1024:05d}-of-01024.json.gz"}
    stream = load_dataset("allenai/c4", data_files=data_files, split="train", streaming=True)
    return stream.shuffle(seed=int(seed), buffer_size=int(buffer_size))


def build_c4_calibration(
    *,
    output: str | Path,
    samples: int = 128,
    seed: int = 0,
    text_field: str = "text",
    min_chars: int = 64,
    buffer_size: int = 10_000,
) -> Path:
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")

    output = Path(output)
    ensure_dir(output.parent)

    source_mode = os.environ.get("C4_SOURCE_MODE", "direct")
    if source_mode == "datasets":
        stream = _iter_c4_dataset_rows(seed, buffer_size)
        source_dataset = "allenai/c4/en/train via datasets streaming"
    else:
        try:
            stream = _iter_c4_direct_rows(seed)
            source_dataset = _c4_direct_url(seed)
        except Exception:
            if source_mode == "direct":
                raise
            stream = _iter_c4_dataset_rows(seed, buffer_size)
            source_dataset = "allenai/c4/en/train via datasets streaming"

    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for index, row in enumerate(stream):
        text = _nonempty_text(dict(row), text_field)
        if len(text) < int(min_chars):
            continue
        rows.append(
            {
                "id": f"c4-seed{seed}-{len(rows):05d}",
                "dataset": "c4",
                "source_dataset": source_dataset,
                "text": text,
                "seed": int(seed),
                "stream_index_hint": int(index),
                "tie_breaker": rng.random(),
            }
        )
        if len(rows) >= int(samples):
            break

    if len(rows) != int(samples):
        raise RuntimeError(f"Expected {samples} C4 rows, collected {len(rows)}")

    write_jsonl(output, rows)
    write_json(
        output.with_suffix(".summary.json"),
        {
            "schema": "c4_calibration_artifact_v1",
            "dataset": "allenai/c4",
            "dataset_config": "en",
            "split": "train",
            "source_dataset": source_dataset,
            "source_mode": source_mode,
            "streaming": True,
            "samples": len(rows),
            "seed": int(seed),
            "text_field": text_field,
            "min_chars": int(min_chars),
            "buffer_size": int(buffer_size),
            "output": str(output),
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic C4 JSONL for Wanda calibration.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    args = parser.parse_args()
    path = build_c4_calibration(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        text_field=args.text_field,
        min_chars=args.min_chars,
        buffer_size=args.buffer_size,
    )
    print(path)


if __name__ == "__main__":
    main()
