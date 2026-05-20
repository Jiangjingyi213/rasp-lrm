from __future__ import annotations

from .answer_match import extract_answer


def answer_flipped(baseline_text: str, counterfactual_text: str) -> bool:
    return extract_answer(baseline_text) != extract_answer(counterfactual_text)
