from __future__ import annotations

import json
import random
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
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
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


def _prompt_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("content"):
                return str(item["content"])
            if isinstance(item, str):
                return item
    if isinstance(value, dict) and value.get("content"):
        return str(value["content"])
    return None


def _nested_value(row: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def _normalize_competition_row(
    row: dict[str, Any],
    idx: int,
    split: str,
    dataset_label: str,
) -> dict[str, Any]:
    question = (
        row.get("problem")
        or row.get("question")
        or row.get("prompt")
        or row.get("input")
        or row.get("query")
    )
    question_text = _prompt_content(question)
    answer = _nested_value(
        row,
        "answer",
        "final_answer",
        "target",
        "label",
        "gold",
        "reward_model.ground_truth",
        "extra_info.answer",
        "extra_info.ground_truth",
    )
    if not question_text:
        raise ValueError(f"{dataset_label} row {idx} is missing a problem/question field")
    if answer is None:
        raise ValueError(f"{dataset_label} row {idx} is missing an answer field")
    pairs = _choice_pairs(row.get("choices") or row.get("options"))
    answer_text = str(answer).strip()
    is_choice = bool(re.fullmatch(r"\(?\s*[A-E]\s*\)?", answer_text.upper()))
    if pairs:
        canonical_pairs, label_map = _canonical_choice_labels(pairs)
        gold = _normalize_choice_answer(answer_text, label_map)
        question_text = _format_multiple_choice_question(question_text, canonical_pairs)
        answer_type = "multiple_choice"
        choices = [{"label": label, "text": text} for label, text in canonical_pairs]
    elif is_choice:
        gold = _normalize_choice_answer(answer_text, {letter: letter for letter in "ABCDE"})
        if "boxed" not in question_text:
            question_text = f"{question_text.strip()}\n\nPut only the option letter in \\boxed{{}}."
        answer_type = "multiple_choice"
        choices = None
    else:
        gold = answer_text
        if "boxed" not in question_text:
            question_text = f"{question_text.strip()}\n\nPut the final answer in \\boxed{{}}."
        answer_type = "math"
        choices = None
    output = {
        "id": row.get("id") or row.get("unique_id") or f"{dataset_label}-{split}-{idx}",
        "dataset": dataset_label,
        "source_dataset": row.get("source_dataset") or row.get("data_source") or dataset_label,
        "question": question_text,
        "gold": gold,
        "answer_type": answer_type,
    }
    if choices is not None:
        output["choices"] = choices
    return output


def load_competition_hf(config: dict[str, Any], dataset_label: str, default_name: str) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", default_name)
    dataset_config = config.get("dataset_config") or config.get("config_name")
    split = config.get("split", "train")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    args = [name_or_path]
    if dataset_config:
        args.append(str(dataset_config))
    dataset, used_split = _load_dataset_with_split_fallback(args, split, config)
    rows = [
        _normalize_competition_row(dict(row), i, used_split, dataset_label)
        for i, row in enumerate(dataset)
    ]
    return slice_rows(rows, limit, offset)


def _load_dataset_with_split_fallback(
    args: list[str],
    split: str,
    config: dict[str, Any],
    **kwargs: Any,
):
    try:
        return load_dataset(*args, split=split, **kwargs), split
    except ValueError as exc:
        if "Unknown split" not in str(exc):
            raise
        fallbacks = config.get("split_fallbacks") or ["test", "validation", "train"]
        tried = {split}
        for fallback in fallbacks:
            fallback = str(fallback)
            if fallback in tried:
                continue
            tried.add(fallback)
            try:
                return load_dataset(*args, split=fallback, **kwargs), fallback
            except ValueError as fallback_exc:
                if "Unknown split" not in str(fallback_exc):
                    raise
                continue
        raise


def load_competition_local(
    path: str | Path,
    dataset_label: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            rows.append(_normalize_competition_row(item, i, "local", dataset_label))
    return slice_rows(rows, limit, offset)


def _choice_pairs(choices: Any) -> list[tuple[str, str]]:
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        if labels or texts:
            return [(str(label).strip(), str(text).strip()) for label, text in zip(labels, texts)]
        output = []
        for key, value in choices.items():
            if str(key).lower() in {"label", "labels", "text", "texts"}:
                continue
            if isinstance(value, (str, int, float)):
                output.append((str(key).strip(), str(value).strip()))
        return output
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


def _format_multiple_select_question(question: str, pairs: list[tuple[str, str]]) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in pairs)
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{options}\n\n"
        "If multiple options are correct, put all correct option letters in alphabetical order in \\boxed{}."
    )


def _serialize_jsonish(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_livecodebench_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = (
        row.get("question_content")
        or row.get("question")
        or row.get("prompt")
        or row.get("content")
        or row.get("description")
    )
    if not question:
        raise ValueError(f"LiveCodeBench row {idx} is missing a question/content field")
    title = row.get("question_title") or row.get("title") or ""
    starter_code = row.get("starter_code") or row.get("starter") or ""
    prompt_parts = []
    if title:
        prompt_parts.append(f"Title: {title}")
    prompt_parts.append(str(question).strip())
    if starter_code:
        prompt_parts.append(f"Starter code:\n```python\n{str(starter_code).strip()}\n```")
    prompt_parts.append(
        "Write a complete Python solution. Put the final code in one fenced Python code block."
    )
    gold = (
        row.get("canonical_solution")
        or row.get("reference_solution")
        or row.get("solution")
        or row.get("answer")
        or row.get("code")
        or ""
    )
    if not gold:
        tests = {
            "public_test_cases": row.get("public_test_cases"),
            "private_test_cases": row.get("private_test_cases"),
            "generated_test_cases": row.get("generated_test_cases"),
            "metadata": row.get("metadata"),
        }
        gold = "__LCB_TESTS__" + _serialize_jsonish({k: v for k, v in tests.items() if v not in (None, "")})
    return {
        "id": (
            row.get("question_id")
            or row.get("id")
            or row.get("problem_id")
            or f"livecodebench-{split}-{idx}"
        ),
        "dataset": "livecodebench",
        "source_dataset": "livecodebench/code_generation_lite",
        "question": "\n\n".join(prompt_parts),
        "gold": str(gold),
        "answer_type": "code_generation",
        "grading_note": (
            "code_generation_lite_exact_if_reference_solution_else_unexecuted_tests_available"
        ),
    }


def load_livecodebench_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "livecodebench/code_generation_lite")
    dataset_config = config.get("dataset_config") or config.get("config_name")
    split = config.get("split", "test")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    args = [name_or_path]
    if dataset_config:
        args.append(str(dataset_config))
    dataset, used_split = _load_dataset_with_split_fallback(args, split, config)
    rows = [_normalize_livecodebench_row(dict(row), i, used_split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def load_livecodebench_local(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(_normalize_livecodebench_row(json.loads(line), i, "local"))
    return slice_rows(rows, limit, offset)


def _normalize_jeebench_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = (
        row.get("question")
        or row.get("problem")
        or row.get("prompt")
        or row.get("input")
        or row.get("Question")
    )
    if not question:
        raise ValueError(f"JEEBench row {idx} is missing a question/problem field")
    answer = _nested_value(
        row,
        "answer",
        "final_answer",
        "target",
        "label",
        "gold",
        "Answer",
        "correct_answer",
        "correct option",
    )
    if answer is None:
        raise ValueError(f"JEEBench row {idx} is missing an answer field")
    pairs = _choice_pairs(row.get("choices") or row.get("options") or row.get("Options"))
    answer_text = str(answer).strip()
    answer_kind = str(
        row.get("answer_type")
        or row.get("type")
        or row.get("question_type")
        or row.get("Question Type")
        or ""
    ).lower()
    compact_letters = re.sub(r"[^A-Za-z]", "", answer_text).upper()
    is_multiple_select = (
        "multi" in answer_kind
        or "multiple correct" in answer_kind
        or bool(re.fullmatch(r"[A-E]{2,}", compact_letters))
    )
    is_single_choice = bool(re.fullmatch(r"[A-E]", compact_letters))
    if pairs:
        canonical_pairs, label_map = _canonical_choice_labels(pairs)
        if is_multiple_select:
            labels = [
                _normalize_choice_answer(letter, label_map)
                for letter in compact_letters
            ]
            gold = "".join(sorted(set(labels)))
            question_text = _format_multiple_select_question(str(question), canonical_pairs)
            normalized_type = "multiple_select"
        else:
            gold = _normalize_choice_answer(answer_text, label_map)
            question_text = _format_multiple_choice_question(str(question), canonical_pairs)
            normalized_type = "multiple_choice"
        choices = [{"label": label, "text": text} for label, text in canonical_pairs]
    elif is_multiple_select:
        gold = "".join(sorted(set(compact_letters)))
        question_text = (
            f"{str(question).strip()}\n\n"
            "Put all correct option letters in alphabetical order in \\boxed{}."
        )
        normalized_type = "multiple_select"
        choices = None
    elif is_single_choice:
        gold = compact_letters
        question_text = (
            f"{str(question).strip()}\n\n"
            "Put only the option letter in \\boxed{}."
        )
        normalized_type = "multiple_choice"
        choices = None
    else:
        gold = answer_text
        question_text = (
            f"{str(question).strip()}\n\n"
            "Put the final answer in \\boxed{}."
        )
        normalized_type = "math"
        choices = None
    output = {
        "id": row.get("id") or row.get("question_id") or row.get("problem_id") or f"jeebench-{split}-{idx}",
        "dataset": "jeebench",
        "source_dataset": "daman1209arora/jeebench",
        "question": question_text,
        "gold": gold,
        "answer_type": normalized_type,
    }
    if choices is not None:
        output["choices"] = choices
    for key in ("subject", "chapter", "year", "paper"):
        if row.get(key) not in (None, ""):
            output[key] = row[key]
    return output


def load_jeebench_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "daman1209arora/jeebench")
    dataset_config = config.get("dataset_config") or config.get("config_name")
    split = config.get("split", "test")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    args = [name_or_path]
    if dataset_config:
        args.append(str(dataset_config))
    dataset, used_split = _load_dataset_with_split_fallback(args, split, config)
    rows = [_normalize_jeebench_row(dict(row), i, used_split) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def load_jeebench_local(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(_normalize_jeebench_row(json.loads(line), i, "local"))
    return slice_rows(rows, limit, offset)


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
    name_or_path = config.get("name_or_path", "allenai/ai2_arc")
    dataset_config = config.get("dataset_config") or config.get("config_name") or default_config
    split = config.get("split", "test")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    try:
        dataset = load_dataset(name_or_path, dataset_config, split=split)
    except Exception:
        if str(name_or_path) == "ai2_arc":
            dataset = load_dataset("allenai/ai2_arc", dataset_config, split=split)
        elif str(name_or_path) == "allenai/ai2_arc":
            dataset = load_dataset("ai2_arc", dataset_config, split=split)
        else:
            raise
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


def _normalize_gpqa_row(row: dict[str, Any], idx: int, split: str, seed: int = 17) -> dict[str, Any]:
    question = (
        row.get("Question")
        or row.get("question")
        or row.get("problem")
        or row.get("input")
        or row.get("prompt")
    )
    if not question:
        raise ValueError(f"GPQA row {idx} is missing a question field")
    pairs = _choice_pairs(row.get("choices") or row.get("options"))
    answer = row.get("answer") or row.get("label") or row.get("target")
    if pairs:
        canonical_pairs, label_map = _canonical_choice_labels(pairs)
        gold = _normalize_choice_answer(answer, label_map)
    else:
        correct = row.get("Correct Answer") or row.get("correct_answer") or row.get("correct")
        incorrects = [
            row.get("Incorrect Answer 1") or row.get("incorrect_answer_1"),
            row.get("Incorrect Answer 2") or row.get("incorrect_answer_2"),
            row.get("Incorrect Answer 3") or row.get("incorrect_answer_3"),
        ]
        if correct is None or any(value is None for value in incorrects):
            raise ValueError(f"GPQA row {idx} is missing answer choices")
        shuffled = [(str(correct).strip(), True)] + [
            (str(value).strip(), False) for value in incorrects
        ]
        random.Random(f"{seed}:{idx}").shuffle(shuffled)
        canonical_pairs = [
            (chr(ord("A") + choice_index), text)
            for choice_index, (text, _is_correct) in enumerate(shuffled)
        ]
        gold = next(
            chr(ord("A") + choice_index)
            for choice_index, (_text, is_correct) in enumerate(shuffled)
            if is_correct
        )
    return {
        "id": row.get("id") or row.get("Record ID") or f"gpqa_diamond-{split}-{idx}",
        "dataset": "gpqa_diamond",
        "source_dataset": "gpqa",
        "question": _format_multiple_choice_question(str(question), canonical_pairs),
        "gold": gold,
        "answer_type": "multiple_choice",
        "choices": [{"label": label, "text": text} for label, text in canonical_pairs],
    }


def load_gpqa_hf(config: dict[str, Any]) -> list[dict[str, Any]]:
    name_or_path = config.get("name_or_path", "Idavidrein/gpqa")
    dataset_config = config.get("dataset_config") or config.get("config_name") or "gpqa_diamond"
    split = config.get("split", "train")
    limit = _as_int_or_none(config.get("limit"))
    offset = int(config.get("offset", 0))
    seed = int(config.get("choice_seed", 17))
    dataset, used_split = _load_gpqa_dataset_with_fallback(name_or_path, dataset_config, split, config)
    rows = [_normalize_gpqa_row(dict(row), i, used_split, seed=seed) for i, row in enumerate(dataset)]
    return slice_rows(rows, limit, offset)


def _load_gpqa_dataset_with_fallback(
    name_or_path: str,
    dataset_config: str | None,
    split: str,
    config: dict[str, Any],
):
    data_files = config.get("data_files")
    candidates: list[tuple[list[str], dict[str, Any], str]] = []
    if data_files:
        candidates.append(([name_or_path], {"data_files": data_files}, split))
    elif dataset_config:
        candidates.append(([name_or_path, str(dataset_config)], {}, split))
    else:
        candidates.append(([name_or_path], {}, split))

    # The canonical Idavidrein/gpqa repository is gated. This public mirror
    # contains the same simple-evals GPQA CSV layout and works with HF mirrors.
    candidates.append(
        (
            ["zai-org/glm-simple-evals-dataset"],
            {"data_files": "gpqa/gpqa_diamond.csv"},
            "train",
        )
    )
    # Last resort for environments that can reach Azure directly, matching
    # OpenAI simple-evals' reference GPQA loader.
    candidates.append(
        (
            ["csv"],
            {"data_files": "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"},
            "train",
        )
    )

    errors = []
    for args, kwargs, candidate_split in candidates:
        try:
            return _load_dataset_with_split_fallback(args, candidate_split, config, **kwargs)
        except (FileNotFoundError, PermissionError, ValueError, ConnectionError) as exc:
            errors.append(f"{args}: {type(exc).__name__}: {exc}")
            continue
    raise FileNotFoundError("Unable to load GPQA Diamond from any configured source:\n" + "\n".join(errors))


def load_gpqa_local(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            rows.append(_normalize_gpqa_row(item, i, "local"))
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
    if name in {"aime2025", "aime25"}:
        if config.get("path"):
            return load_competition_local(config["path"], "aime2025", limit, offset)
        return load_competition_hf(config, "aime2025", "math-ai/aime25")
    if name in {"amc", "amc2023", "amc23"}:
        if config.get("path"):
            return load_competition_local(config["path"], "amc2023", limit, offset)
        return load_competition_hf(config, "amc2023", "zwhe99/amc23")
    if name in {"gpqa", "gpqa_diamond", "gpqa-diamond"}:
        if config.get("path"):
            return load_gpqa_local(config["path"], limit, offset)
        return load_gpqa_hf(config)
    if name in {"livecodebench", "lcb", "lcb_lite", "livecodebench_lite"}:
        if config.get("path"):
            return load_livecodebench_local(config["path"], limit, offset)
        return load_livecodebench_hf(config)
    if name in {"jeebench", "jee", "jee_bench"}:
        if config.get("path"):
            return load_jeebench_local(config["path"], limit, offset)
        return load_jeebench_hf(config)
    if name in {"arc_easy", "arc-easy", "ai2_arc_easy"}:
        merged = dict(config)
        merged.setdefault("dataset", "arc_easy")
        merged.setdefault("name_or_path", "ai2_arc")
        merged.setdefault("dataset_config", "ARC-Easy")
        return load_arc_hf(merged)
    if name in {"arc_challenge", "arc-challenge", "ai2_arc_challenge"}:
        merged = dict(config)
        merged.setdefault("dataset", "arc_challenge")
        merged.setdefault("name_or_path", "ai2_arc")
        merged.setdefault("dataset_config", "ARC-Challenge")
        return load_arc_hf(merged)
    if name in {"bbh", "bbh_selected", "big_bench_hard"}:
        return load_bbh_selected_hf(config)
    raise ValueError(f"Unsupported dataset: {name}")
