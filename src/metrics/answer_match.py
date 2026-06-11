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
SYMBOLIC_HINT_RE = re.compile(r"[A-Za-zπ]|\\pi|[()=]")


def extract_answer(text: str) -> str:
    text = text or ""
    boxed = extract_last_boxed(text)
    if boxed:
        return normalize_answer_span(boxed, allow_numeric_fallback=False)
    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_answer_span(match.group(1), allow_numeric_fallback=True)
    if looks_like_short_answer(text):
        return normalize_answer_span(text, allow_numeric_fallback=True)
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return clean_answer(numbers[-1])
    return clean_answer(text.strip().splitlines()[-1] if text.strip() else "")


def normalize_answer_span(answer: str, allow_numeric_fallback: bool = True) -> str:
    for marker in ANSWER_STOP_MARKERS:
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    answer = clean_answer(answer)
    fractions = SIMPLE_FRACTION_RE.findall(answer)
    if fractions and is_numeric_like(answer):
        return fractions[-1]
    latex_numbers = LATEX_NUMBER_RE.findall(answer)
    if latex_numbers:
        return clean_answer(latex_numbers[-1])
    numbers = NUMBER_RE.findall(answer)
    if numbers and allow_numeric_fallback and (is_numeric_like(answer) or is_number_with_units(answer)):
        return clean_answer(numbers[-1])
    return answer


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", answer)
    answer = FRAC_RE.sub(r"\1/\2", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\", " ")
    answer = re.sub(r"\b(text|mathrm)\s*\{([^{}]+)\}", r"\2", answer)
    answer = re.sub(r"[*_`#]", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    answer = normalize_symbolic_spacing(answer)
    answer = answer.strip().rstrip(".")
    return answer


def normalize_symbolic_spacing(answer: str) -> str:
    answer = re.sub(r"\s*([(),=+\-*/])\s*", r"\1", answer)
    return answer.strip()


def looks_like_short_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped and len(stripped) > 120:
        return False
    return len(stripped) <= 200


def is_numeric_like(answer: str) -> bool:
    answer = clean_answer_for_numeric_check(answer)
    return bool(answer) and not SYMBOLIC_HINT_RE.search(answer) and bool(re.fullmatch(r"[-+0-9./ ]+", answer))


def is_number_with_units(answer: str) -> bool:
    answer = clean_answer_for_numeric_check(answer)
    if any(ch in answer for ch in "()=+*/") or "-" in answer:
        return False
    return bool(NUMBER_RE.search(answer))


def clean_answer_for_numeric_check(answer: str) -> str:
    answer = answer.replace(" ", "")
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
    if not is_numeric_like(value):
        return None
    try:
        return Decimal(clean_answer(value))
    except (InvalidOperation, ValueError):
        return None


def _as_fraction(value: str) -> Fraction | None:
    if not is_numeric_like(value):
        return None
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
    if pred.lower() == gold_answer.lower():
        return True
    # MATH contains short categorical answers such as person names. A model
    # may place the correct category in a final-answer sentence without boxing
    # it, so accept a single alphabetic gold token only as a whole word.
    if re.fullmatch(r"[A-Za-z]+", gold_answer):
        return bool(re.search(rf"\b{re.escape(gold_answer)}\b", pred, re.IGNORECASE))
    return False
