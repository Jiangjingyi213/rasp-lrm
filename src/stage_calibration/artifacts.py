from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hash(rows: Iterable[dict[str, Any]], fields: tuple[str, ...] = ("id", "question")) -> str:
    payload = [{field: row.get(field) for field in fields} for row in rows]
    return stable_hash(payload)


def assert_metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Artifact metadata mismatch: {mismatches}")
