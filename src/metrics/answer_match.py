from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


FINAL_PATTERNS = [
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
]
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    text = text or ""
    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return clean_answer(match.group(1))
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return clean_answer(numbers[-1])
    return clean_answer(text.strip().splitlines()[-1] if text.strip() else "")


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", answer)
    answer = answer.strip().rstrip(".")
    return answer


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(clean_answer(value))
    except (InvalidOperation, ValueError):
        return None


def answer_match(prediction: str, gold: str) -> bool:
    pred = extract_answer(prediction)
    gold_answer = extract_answer(gold)
    pred_num = _as_decimal(pred)
    gold_num = _as_decimal(gold_answer)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return pred.lower() == gold_answer.lower()
