from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


DEFAULT_ALLOWED_SOURCES = (
    "Orca-Math",
    "olympiads",
    "aops_forum",
    "HARP",
    "Big-Math-Reformulated",
)
DEFAULT_EXCLUDED_SOURCES = ("GSM8k", "MATH", "Omni-MATH", "amc_aime", "cn_k12")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def char_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = normalize_text(text).replace(" ", "")
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def source_name(row: dict[str, Any]) -> str:
    for key in ("source", "source_dataset", "dataset", "origin"):
        if row.get(key):
            return str(row[key])
    return ""


def normalize_big_math_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = row.get("problem") or row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("final_answer") or row.get("target")
    if not question or answer is None:
        raise ValueError("Big-Math row is missing problem/question or answer")
    source = source_name(row)
    return {
        "id": str(row.get("id") or row.get("unique_id") or f"big-math-{index}"),
        "dataset": "big_math_rl_verified",
        "source": source,
        "domain": str(row.get("domain") or row.get("type") or row.get("subject") or "unknown"),
        "question": str(question),
        "gold": str(answer),
    }


def source_allowed(
    source: str,
    allowed: Iterable[str] = DEFAULT_ALLOWED_SOURCES,
    excluded: Iterable[str] = DEFAULT_EXCLUDED_SOURCES,
) -> bool:
    normalized = normalize_text(source)
    excluded_values = {normalize_text(value) for value in excluded}
    allowed_values = {normalize_text(value) for value in allowed}
    return normalized in allowed_values and normalized not in excluded_values


def decontaminate(
    candidates: Iterable[dict[str, Any]],
    protected_rows: Iterable[dict[str, Any]],
    threshold: float = 0.80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protected = []
    exact = {}
    for row in protected_rows:
        normalized = normalize_text(row["question"])
        exact[normalized] = str(row.get("id", "unknown"))
        protected.append((str(row.get("id", "unknown")), char_ngrams(row["question"])))
    kept = []
    excluded = []
    seen: dict[str, str] = {}
    for row in candidates:
        normalized = normalize_text(row["question"])
        reason = None
        nearest_id = None
        similarity = 0.0
        if normalized in seen:
            reason = "duplicate_within_pool"
            nearest_id = seen[normalized]
            similarity = 1.0
        elif normalized in exact:
            reason = "exact_test_overlap"
            nearest_id = exact[normalized]
            similarity = 1.0
        else:
            grams = char_ngrams(row["question"])
            for protected_id, protected_grams in protected:
                value = jaccard(grams, protected_grams)
                if value > similarity:
                    similarity = value
                    nearest_id = protected_id
            if similarity >= threshold:
                reason = "near_test_overlap"
        if reason:
            excluded.append(
                {
                    **row,
                    "exclusion_reason": reason,
                    "nearest_protected_id": nearest_id,
                    "similarity": similarity,
                }
            )
            continue
        seen[normalized] = str(row["id"])
        kept.append(row)
    return kept, excluded


def stratified_split(
    rows: list[dict[str, Any]], calibration_size: int, dev_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if calibration_size + dev_size > len(rows):
        raise ValueError("Not enough eligible rows for disjoint calibration/dev split")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("source", "")), str(row.get("domain", "unknown")))
        groups.setdefault(key, []).append(row)
    ordered = []
    for key in sorted(groups):
        values = groups[key]
        rng.shuffle(values)
        ordered.extend(values)
    rng.shuffle(ordered)
    calibration = ordered[:calibration_size]
    dev = ordered[calibration_size : calibration_size + dev_size]
    if {row["id"] for row in calibration} & {row["id"] for row in dev}:
        raise ValueError("Calibration/dev split overlap")
    return calibration, dev


def source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source", "")) for row in rows))
