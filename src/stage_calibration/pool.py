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


def strip_embedded_answer(question: str) -> str:
    markers = list(
        re.finditer(
            r"(?im)^\s*(?:answer|final\s+answer|答案)\s*[:：]",
            str(question),
        )
    )
    if not markers:
        return str(question).strip()
    return str(question)[: markers[-1].start()].strip()


def normalize_big_math_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = row.get("problem") or row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("final_answer") or row.get("target") or row.get("solution")
    if not question or answer is None:
        raise ValueError("Big-Math row is missing problem/question or answer")
    source = source_name(row)
    normalized = {
        "id": str(row.get("id") or row.get("unique_id") or row.get("uuid") or f"math-pool-{index}"),
        "dataset": str(row.get("dataset") or "math_calibration_pool"),
        "source": source,
        "domain": str(
            row.get("domain")
            or row.get("problem_type")
            or row.get("type")
            or row.get("subject")
            or "unknown"
        ),
        "question": strip_embedded_answer(str(question)),
        "gold": str(answer),
        "answer_type": "math",
    }
    if row.get("solution"):
        normalized["solution"] = str(row["solution"])
    if row.get("level"):
        normalized["level"] = str(row["level"])
    return normalized


def allocate_source_targets(total: int, sources: list[dict[str, Any]]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not sources:
        raise ValueError("At least one source is required")
    weights = [float(source.get("quota", source.get("weight", 1.0))) for source in sources]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Source quotas/weights must be non-negative and sum to a positive value")
    raw = [total * weight / sum(weights) for weight in weights]
    targets = [int(value) for value in raw]
    remainder = total - sum(targets)
    order = sorted(range(len(sources)), key=lambda index: (raw[index] - targets[index], -index), reverse=True)
    for index in order[:remainder]:
        targets[index] += 1
    return targets


def normalize_calibration_pool_row(
    row: dict[str, Any],
    index: int,
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    adapter = str(source_cfg.get("adapter", "math_word_problem"))
    if adapter in {"math", "word_problem", "math_word_problem", "big_math", "orca_math", "gsm8k_train", "math_train"}:
        normalized = normalize_big_math_row(row, index)
    elif adapter in {"multiple_choice", "aqua_rat", "openbookqa", "commonsenseqa"}:
        normalized = normalize_multiple_choice_row(row, index, source_cfg)
    else:
        raise ValueError(f"Unsupported calibration pool adapter: {adapter}")
    source = str(
        source_cfg.get("source")
        or source_cfg.get("default_source")
        or source_cfg.get("name")
        or normalized.get("source")
        or adapter
    )
    normalized["source"] = source
    normalized["source_dataset"] = str(source_cfg.get("name_or_path", normalized.get("dataset", source)))
    normalized["dataset"] = str(source_cfg.get("name", normalized.get("dataset", source)))
    if source_cfg.get("default_domain"):
        normalized["domain"] = str(source_cfg["default_domain"])
    normalized["id"] = f"{source}:{normalized['id']}"
    return normalized


def normalize_multiple_choice_row(
    row: dict[str, Any],
    index: int,
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    question = _extract_question(row)
    labels, texts = _extract_choices(row)
    answer = row.get("answerKey") or row.get("answer_key") or row.get("correct") or row.get("answer")
    if answer is None:
        raise ValueError("Multiple-choice row is missing answer key")
    gold = _normalize_choice_answer(str(answer), labels, texts)
    if not question or not labels or not texts:
        raise ValueError("Multiple-choice row is missing question or choices")
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    formatted_question = (
        f"{str(question).strip()}\n\n"
        f"Options:\n{options}\n\n"
        "Choose the single best option. In the final stage, output only the option letter in \\boxed{}."
    )
    return {
        "id": str(row.get("id") or row.get("qid") or row.get("question_id") or f"mc-pool-{index}"),
        "dataset": str(source_cfg.get("name") or row.get("dataset") or "multiple_choice_calibration_pool"),
        "source": str(source_cfg.get("source") or source_cfg.get("default_source") or source_cfg.get("name") or ""),
        "domain": str(source_cfg.get("default_domain") or row.get("domain") or row.get("subject") or "non_math_reasoning"),
        "question": formatted_question,
        "gold": gold,
        "answer_type": "multiple_choice",
        "choice_labels": labels,
        "choice_texts": texts,
    }


def _extract_question(row: dict[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, dict):
        return str(question.get("stem") or question.get("question") or question.get("text") or "")
    return str(row.get("question_stem") or row.get("stem") or row.get("prompt") or question or "")


def _extract_choices(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = row.get("choices") or row.get("options")
    labels: list[str] = []
    texts: list[str] = []
    if isinstance(choices, dict):
        raw_labels = choices.get("label") or choices.get("labels") or []
        raw_texts = choices.get("text") or choices.get("texts") or []
        labels = [str(value).strip().upper() for value in raw_labels]
        texts = [str(value).strip() for value in raw_texts]
    elif isinstance(choices, list):
        for offset, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = str(choice.get("label") or chr(ord("A") + offset)).strip().upper()
                text = str(choice.get("text") or choice.get("content") or "").strip()
            else:
                label, text = _parse_choice_string(str(choice), offset)
            labels.append(label)
            texts.append(text)
    if not labels and texts:
        labels = [chr(ord("A") + index) for index in range(len(texts))]
    if labels and len(labels) != len(texts):
        raise ValueError("Choice labels/texts length mismatch")
    return labels, texts


def _parse_choice_string(choice: str, offset: int) -> tuple[str, str]:
    match = re.match(r"^\s*([A-Za-z])\s*[\).:]\s*(.*)$", choice)
    if match:
        return match.group(1).upper(), match.group(2).strip()
    return chr(ord("A") + offset), choice.strip()


def _normalize_choice_answer(answer: str, labels: list[str], texts: list[str]) -> str:
    cleaned = str(answer).strip().upper()
    match = re.match(r"^\s*([A-Z])\s*[\).:]?\s*", cleaned)
    if match and match.group(1) in labels:
        return match.group(1)
    for label, text in zip(labels, texts):
        if cleaned == str(text).strip().upper():
            return label
    if cleaned in labels:
        return cleaned
    raise ValueError(f"Multiple-choice answer `{answer}` does not match labels {labels}")


def _source_pattern_matches(source: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    if "*" in pattern:
        regex = "^" + re.escape(pattern.lower()).replace("\\*", ".*") + "$"
        return re.search(regex, source.lower()) is not None
    return normalize_text(source) == normalize_text(pattern)


def source_allowed(
    source: str,
    allowed: Iterable[str] = DEFAULT_ALLOWED_SOURCES,
    excluded: Iterable[str] = DEFAULT_EXCLUDED_SOURCES,
) -> bool:
    source = str(source)
    return any(_source_pattern_matches(source, str(value)) for value in allowed) and not any(
        _source_pattern_matches(source, str(value)) for value in excluded
    )


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
