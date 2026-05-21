from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


FINAL_PATTERNS = [
    re.compile(r"final answer\s*[:：]?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"####\s*([^\n]+)"),
]
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
LATEX_NUMBER_RE = re.compile(r"\$+\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\$+")
FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
SIMPLE_FRACTION_RE = re.compile(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?")
ANSWER_STOP_MARKERS = ("Human:", "User:", "Assistant:", "Problem:")


def extract_answer(text: str) -> str:
    text = text or ""
    boxed = extract_last_boxed(text)
    if boxed:
        return normalize_answer_span(boxed)
    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_answer_span(match.group(1))
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return clean_answer(numbers[-1])
    return clean_answer(text.strip().splitlines()[-1] if text.strip() else "")


def normalize_answer_span(answer: str) -> str:
    for marker in ANSWER_STOP_MARKERS:
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    answer = clean_answer(answer)
    fractions = SIMPLE_FRACTION_RE.findall(answer)
    if fractions:
        return fractions[-1]
    latex_numbers = LATEX_NUMBER_RE.findall(answer)
    if latex_numbers:
        return clean_answer(latex_numbers[-1])
    numbers = NUMBER_RE.findall(answer)
    if numbers:
        return clean_answer(numbers[-1])
    return answer


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", answer)
    answer = FRAC_RE.sub(r"\1/\2", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\", " ")
    answer = re.sub(r"\b(text|mathrm)\s*\{([^{}]+)\}", r"\2", answer)
    answer = re.sub(r"[*_`#]", "", answer)
    answer = answer.strip().rstrip(".")
    return answer


def extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    starts = [m.start() for m in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        pos = start + len(marker)
        depth = 1
        chars = []
        while pos < len(text):
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars)
                chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
    boxed = BOXED_RE.findall(text)
    return boxed[-1] if boxed else None


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(clean_answer(value))
    except (InvalidOperation, ValueError):
        return None


def _as_fraction(value: str) -> Fraction | None:
    try:
        cleaned = clean_answer(value).replace(" ", "")
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def answer_match(prediction: str, gold: str) -> bool:
    pred = extract_answer(prediction)
    gold_answer = extract_answer(gold)
    pred_num = _as_decimal(pred)
    gold_num = _as_decimal(gold_answer)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    pred_frac = _as_fraction(pred)
    gold_frac = _as_fraction(gold_answer)
    if pred_frac is not None and gold_frac is not None:
        return pred_frac == gold_frac
    return pred.lower() == gold_answer.lower()
