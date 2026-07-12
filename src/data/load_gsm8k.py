from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # Keep normalization tests usable in lightweight local envs.
    def load_dataset(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("The `datasets` package is required to load Hugging Face datasets")


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


def _as_int_or_none(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    return int(value)


def load_gsm8k(split: str = "test", limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=split)
    rows = [_normalize_gsm8k(dict(row), i, split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def _normalize_math500(row: dict[str, Any], idx: int, split: str, dataset_label: str = "math500") -> dict[str, Any]:
    question = row.get("problem") or row.get("question")
    answer = row.get("answer") or row.get("final_answer")
    solution = row.get("solution") or row.get("gold") or ""
    if not question:
        raise ValueError(f"MATH500 row {idx} is missing a problem/question field")
    return {
        "id": row.get("id") or row.get("unique_id") or f"{dataset_label}-{split}-{idx}",
        "dataset": dataset_label,
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
    dataset_label = str(config.get("dataset_label", "math500"))
    dataset = load_dataset(name_or_path, split=split)
    rows = [_normalize_math500(dict(row), i, split, dataset_label) for i, row in enumerate(dataset)]
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


def _choice_pairs(choices: Any) -> list[tuple[str, str]]:
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        return [(str(label).strip(), str(text).strip()) for label, text in zip(labels, texts)]
    if isinstance(choices, list):
        pairs = []
        for index, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = choice.get("label") or choice.get("labels") or chr(ord("A") + index)
                text = choice.get("text") or choice.get("content") or choice.get("value") or ""
            else:
                label = chr(ord("A") + index)
                text = choice
            pairs.append((str(label).strip(), str(text).strip()))
        return pairs
    return []


def _canonical_choice_labels(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    output = []
    label_map: dict[str, str] = {}
    for index, (raw_label, text) in enumerate(pairs):
        canonical = chr(ord("A") + index)
        output.append((canonical, text))
        if raw_label:
            label_map[raw_label.upper()] = canonical
        label_map[canonical] = canonical
        label_map[str(index + 1)] = canonical
    return output, label_map


def _normalize_choice_answer(answer: Any, label_map: dict[str, str]) -> str:
    raw = str(answer).strip()
    cleaned = raw.strip("()[]{} ").upper()
    return label_map.get(cleaned, cleaned)


def _format_multiple_choice_question(question: str, pairs: list[tuple[str, str]]) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in pairs)
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{options}\n\n"
        "Put only the option letter in \\boxed{}."
    )


def _normalize_arc_row(
    row: dict[str, Any],
    idx: int,
    split: str,
    dataset_label: str,
) -> dict[str, Any]:
    question = row.get("question") or row.get("question_stem") or row.get("input")
    if not question:
        raise ValueError(f"ARC row {idx} is missing a question field")
    pairs, label_map = _canonical_choice_labels(_choice_pairs(row.get("choices")))
    if not pairs:
        raise ValueError(f"ARC row {idx} is missing choices")
    gold = _normalize_choice_answer(row.get("answerKey") or row.get("answer") or row.get("label"), label_map)
    return {
        "id": row.get("id") or f"{dataset_label}-{split}-{idx}",
        "dataset": dataset_label,
        "source_dataset": "ai2_arc",
        "question": _format_multiple_choice_question(str(question), pairs),
        "gold": gold,
        "answer_type": "multiple_choice",
        "choices": [{"label": label, "text": text} for label, text in pairs],
    }


def load_arc_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_label = str(config.get("dataset_label") or config.get("dataset", "arc_easy"))
    default_config = "ARC-Challenge" if "challenge" in dataset_label.lower() else "ARC-Easy"
    name_or_path = config.get("name_or_path", "ai2_arc")
    dataset_config = config.get("dataset_config") or config.get("config_name") or default_config
    split = config.get("split", "test")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    dataset = load_dataset(name_or_path, dataset_config, split=split)
    rows = [_normalize_arc_row(dict(row), i, split, dataset_label) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


_CHOICE_ANSWER_RE = re.compile(r"^\(?\s*([A-Z])\s*\)?$")


def _normalize_bbh_row(row: dict[str, Any], idx: int, split: str, task: str) -> dict[str, Any]:
    question = row.get("input") or row.get("question") or row.get("prompt")
    gold = row.get("target") or row.get("answer") or row.get("label")
    if question is None:
        raise ValueError(f"BBH row {idx} in {task} is missing an input/question field")
    if gold is None:
        raise ValueError(f"BBH row {idx} in {task} is missing a target/answer field")
    gold_text = str(gold).strip()
    match = _CHOICE_ANSWER_RE.match(gold_text.upper())
    if match:
        answer_type = "multiple_choice"
        normalized_gold = match.group(1)
        instruction = "Put only the option letter in \\boxed{}."
    else:
        answer_type = "exact"
        normalized_gold = gold_text
        instruction = "Put the final answer in \\boxed{}."
    return {
        "id": row.get("id") or f"bbh_selected-{task}-{split}-{idx}",
        "dataset": "bbh_selected",
        "source_dataset": "bbh",
        "bbh_task": task,
        "question": f"{str(question).strip()}\n\n{instruction}",
        "gold": normalized_gold,
        "answer_type": answer_type,
    }


def load_bbh_selected_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "lukaemon/bbh")
    split = config.get("split", "test")
    tasks = config.get(
        "selected_tasks",
        [
            "boolean_expressions",
            "date_understanding",
            "disambiguation_qa",
            "logical_deduction_three_objects",
            "logical_deduction_five_objects",
            "tracking_shuffled_objects_three_objects",
        ],
    )
    per_task_limit = _as_int_or_none(config.get("per_task_limit"))
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        dataset = load_dataset(name_or_path, str(task), split=split)
        task_rows = [
            _normalize_bbh_row(dict(row), i, split, str(task))
            for i, row in enumerate(dataset)
        ]
        if per_task_limit is not None:
            task_rows = task_rows[:per_task_limit]
        rows.extend(task_rows)
    return slice_rows(rows, limit, offset)


def load_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    name = config.get("dataset", "gsm8k").lower()
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
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
    if name in {"arc_easy", "arc-easy", "ai2_arc_easy"}:
        merged = dict(config)
        merged.setdefault("dataset", "arc_easy")
        merged.setdefault("dataset_config", "ARC-Easy")
        return load_arc_hf(merged)
    if name in {"arc_challenge", "arc-challenge", "ai2_arc_challenge"}:
        merged = dict(config)
        merged.setdefault("dataset", "arc_challenge")
        merged.setdefault("dataset_config", "ARC-Challenge")
        return load_arc_hf(merged)
    if name in {"bbh", "bbh_selected", "big_bench_hard"}:
        return load_bbh_selected_hf(config)
    raise ValueError(f"Unsupported dataset: {name}")
