from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _normalize_gsm8k(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    return {
        "id": row.get("id", f"gsm8k-{split}-{idx}"),
        "dataset": "gsm8k",
        "question": row["question"],
        "gold": row.get("answer", ""),
    }


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=split)
    rows = [_normalize_gsm8k(dict(row), i, split) for i, row in enumerate(dataset)]
    return rows[:limit] if limit else rows


def load_math500(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            rows.append(
                {
                    "id": item.get("id", f"math500-{i}"),
                    "dataset": "math500",
                    "question": item.get("problem") or item.get("question"),
                    "gold": item.get("answer") or item.get("solution", ""),
                }
            )
    return rows[:limit] if limit else rows


def load_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    name = config.get("dataset", "gsm8k").lower()
    limit = config.get("limit")
    if name == "gsm8k":
        return load_gsm8k(config.get("split", "test"), limit)
    if name == "math500":
        if not config.get("path"):
            raise ValueError("MATH500 requires dataset.path in the experiment config")
        return load_math500(config["path"], limit)
    raise ValueError(f"Unsupported dataset: {name}")
