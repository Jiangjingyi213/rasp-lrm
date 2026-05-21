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


def slice_rows(rows: list[dict[str, Any]], limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def load_gsm8k(split: str = "test", limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=split)
    rows = [_normalize_gsm8k(dict(row), i, split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def _normalize_math500(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = row.get("problem") or row.get("question")
    answer = row.get("answer") or row.get("final_answer")
    solution = row.get("solution") or row.get("gold") or ""
    if not question:
        raise ValueError(f"MATH500 row {idx} is missing a problem/question field")
    return {
        "id": row.get("id") or row.get("unique_id") or f"math500-{split}-{idx}",
        "dataset": "math500",
        "question": question,
        "gold": answer or solution,
        "solution": solution,
    }


def load_math500_local(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            rows.append(_normalize_math500(item, i, "local"))
    return slice_rows(rows, limit, offset)


def load_math500_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "HuggingFaceH4/MATH-500")
    split = config.get("split", "test")
    limit = config.get("limit")
    offset = config.get("offset", 0)
    dataset = load_dataset(name_or_path, split=split)
    rows = [_normalize_math500(dict(row), i, split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def _normalize_aime(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = row.get("problem") or row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("final_answer") or row.get("target")
    solution = row.get("solution") or row.get("rationale") or ""
    if not question:
        raise ValueError(f"AIME row {idx} is missing a problem/question field")
    return {
        "id": row.get("id") or row.get("unique_id") or f"aime2024-{split}-{idx}",
        "dataset": "aime2024",
        "question": question,
        "gold": str(answer if answer is not None else solution),
        "solution": solution,
    }


def load_aime2024_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "HuggingFaceH4/aime_2024")
    split = config.get("split", "train")
    limit = config.get("limit")
    offset = config.get("offset", 0)
    dataset = load_dataset(name_or_path, split=split)
    rows = [_normalize_aime(dict(row), i, split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def load_aime2024_local(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            rows.append(_normalize_aime(item, i, "local"))
    return slice_rows(rows, limit, offset)


def load_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    name = config.get("dataset", "gsm8k").lower()
    limit = config.get("limit")
    offset = config.get("offset", 0)
    if name == "gsm8k":
        return load_gsm8k(config.get("split", "test"), limit, offset)
    if name == "math500":
        if config.get("path"):
            return load_math500_local(config["path"], limit, offset)
        return load_math500_hf(config)
    if name in {"aime", "aime2024"}:
        if config.get("path"):
            return load_aime2024_local(config["path"], limit, offset)
        return load_aime2024_hf(config)
    raise ValueError(f"Unsupported dataset: {name}")
