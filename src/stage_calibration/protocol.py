from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


STAGES = ("setup", "reasoning", "verify", "final")
MARKERS = {
    "setup": "<STAGE_SETUP>",
    "reasoning": "<STAGE_REASONING>",
    "verify": "<STAGE_VERIFY>",
    "final": "<STAGE_FINAL>",
}
MARKER_TO_STAGE = {marker: stage for stage, marker in MARKERS.items()}
STAGE_TAG_RE = re.compile(r"<STAGE_[A-Z_]+>")
ANY_STAGE_TAG_RE = re.compile(r"</?STAGE_[A-Z_]+>")


def explicit_stage_instruction() -> str:
    return (
        "Solve the problem using exactly these four stage markers, each exactly once and in order:\n"
        "<STAGE_SETUP>\n"
        "<STAGE_REASONING>\n"
        "<STAGE_VERIFY>\n"
        "<STAGE_FINAL>\n"
        "Only use these opening markers. Never write closing markers such as </STAGE_SETUP>. "
        "Never restart from <STAGE_SETUP> after <STAGE_FINAL>. "
        "Inside <STAGE_FINAL>, output only the final boxed answer in \\boxed{} and then stop. "
        "Do not write any other <STAGE_...> marker."
    )


def illegal_stage_tag_reason(decoded_text: str) -> str | None:
    for match in ANY_STAGE_TAG_RE.finditer(decoded_text):
        marker = match.group(0)
        if marker.startswith("</"):
            return f"closing_stage_marker:{marker}"
        if marker not in MARKER_TO_STAGE:
            return f"unknown_stage_marker:{marker}"
    return None


def marker_token_sequences(tokenizer) -> dict[str, tuple[tuple[int, ...], ...]]:
    sequences = {}
    for stage, marker in MARKERS.items():
        variants = []
        for prefix in ("", "\n", "\n\n", " "):
            for suffix in ("", "\n", "\n\n", " "):
                token_ids = tuple(
                    int(value)
                    for value in tokenizer(prefix + marker + suffix, add_special_tokens=False).input_ids
                )
                if token_ids and token_ids not in variants:
                    variants.append(token_ids)
            token_ids = tuple(
                int(value)
                for value in tokenizer(prefix + marker, add_special_tokens=False).input_ids
            )
            if token_ids and token_ids not in variants:
                variants.append(token_ids)
        if not variants:
            raise ValueError(f"Tokenizer produced no ids for {marker}")
        sequences[stage] = tuple(variants)
    return sequences


@dataclass
class StageTokenTracker:
    marker_sequences: dict[str, Any]
    generated_ids: list[int] = field(default_factory=list)
    active_stage: str | None = None
    next_stage_index: int = 0
    fallback_reason: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    marker_positions: list[tuple[int, int, str]] = field(default_factory=list)

    def _variants(self, stage: str) -> tuple[tuple[int, ...], ...]:
        value = self.marker_sequences[stage]
        if value and isinstance(value[0], int):
            return (tuple(value),)
        return tuple(tuple(sequence) for sequence in value)

    def feed(self, token_id: int) -> str | None:
        self.generated_ids.append(int(token_id))
        if self.fallback_reason is not None:
            return None
        matched = []
        for stage in self.marker_sequences:
            for sequence in self._variants(stage):
                if len(self.generated_ids) >= len(sequence) and tuple(
                    self.generated_ids[-len(sequence) :]
                ) == sequence:
                    matched.append((stage, sequence))
        if not matched:
            return None
        stage, sequence = matched[0]
        expected = STAGES[self.next_stage_index] if self.next_stage_index < len(STAGES) else None
        if stage != expected:
            self.fallback_reason = f"invalid_stage_transition:{expected}->{stage}"
            self.active_stage = None
            return None
        end = len(self.generated_ids)
        start = end - len(sequence)
        self.marker_positions.append((start, end, stage))
        self.active_stage = stage
        self.next_stage_index += 1
        self.transitions.append({"stage": stage, "generated_tokens": end})
        return stage

    def fallback_dense(self, reason: str) -> None:
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)
        self.active_stage = None

    def finalize(self, decoded_text: str = "") -> dict[str, Any]:
        illegal_reason = illegal_stage_tag_reason(decoded_text)
        if illegal_reason:
            self.fallback_dense(illegal_reason)
        valid = self.fallback_reason is None and self.next_stage_index == len(STAGES)
        if self.fallback_reason is None and not valid:
            self.fallback_reason = f"missing_stage_markers:{self.next_stage_index}/{len(STAGES)}"
        assignments: list[str | None] = [None] * len(self.generated_ids)
        for index, (_start, end, stage) in enumerate(self.marker_positions):
            content_end = (
                self.marker_positions[index + 1][0]
                if index + 1 < len(self.marker_positions)
                else len(self.generated_ids)
            )
            for token_index in range(end, content_end):
                assignments[token_index] = stage
        spans = []
        for start, end, stage in self.marker_positions:
            content_end = next(
                (
                    marker_start
                    for marker_start, _marker_end, _next_stage in self.marker_positions
                    if marker_start >= end
                ),
                len(self.generated_ids),
            )
            spans.append(
                {
                    "stage": stage,
                    "marker_start_token": start,
                    "content_start_token": end,
                    "content_end_token": content_end,
                    "content_tokens": max(0, content_end - end),
                }
            )
        return {
            "valid": valid,
            "fallback_reason": self.fallback_reason,
            "transitions": self.transitions,
            "stage_spans": spans,
            "token_stages": assignments,
        }


def analyze_generated_ids(tokenizer, generated_ids: list[int]) -> dict[str, Any]:
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tracker = StageTokenTracker(marker_token_sequences(tokenizer))
    for token_id in generated_ids:
        tracker.feed(int(token_id))
    result = tracker.finalize(decoded_text)
    if result["valid"]:
        result["detected_by"] = "token_sequence"
        return result
    text_result = analyze_decoded_text_markers(tokenizer, generated_ids, decoded_text)
    if text_result is not None and text_result["valid"]:
        return text_result
    result["detected_by"] = "token_sequence"
    return result


def analyze_decoded_text_markers(
    tokenizer, generated_ids: list[int], decoded_text: str
) -> dict[str, Any] | None:
    if illegal_stage_tag_reason(decoded_text):
        return None
    matches = list(STAGE_TAG_RE.finditer(decoded_text))
    if not matches:
        return None
    markers = [match.group(0) for match in matches]
    if any(marker not in MARKER_TO_STAGE for marker in markers):
        return None
    stages = [MARKER_TO_STAGE[marker] for marker in markers]
    if tuple(stages) != STAGES:
        return None

    marker_positions = []
    for match, stage in zip(matches, stages):
        start = _token_index_at_decoded_char(tokenizer, generated_ids, match.start())
        end = _token_index_at_decoded_char(tokenizer, generated_ids, match.end())
        marker_positions.append((start, max(start, end), stage))

    assignments: list[str | None] = [None] * len(generated_ids)
    spans = []
    for index, (start, end, stage) in enumerate(marker_positions):
        content_end = (
            marker_positions[index + 1][0]
            if index + 1 < len(marker_positions)
            else len(generated_ids)
        )
        for token_index in range(end, content_end):
            assignments[token_index] = stage
        spans.append(
            {
                "stage": stage,
                "marker_start_token": start,
                "content_start_token": end,
                "content_end_token": content_end,
                "content_tokens": max(0, content_end - end),
            }
        )

    return {
        "valid": True,
        "fallback_reason": None,
        "transitions": [
            {"stage": stage, "generated_tokens": end}
            for _start, end, stage in marker_positions
        ],
        "stage_spans": spans,
        "token_stages": assignments,
        "detected_by": "decoded_text",
    }


def _token_index_at_decoded_char(tokenizer, generated_ids: list[int], char_index: int) -> int:
    if char_index <= 0:
        return 0
    cache: dict[int, int] = {}

    def decoded_len(end: int) -> int:
        if end not in cache:
            cache[end] = len(tokenizer.decode(generated_ids[:end], skip_special_tokens=True))
        return cache[end]

    low = 0
    high = len(generated_ids)
    while low < high:
        mid = (low + high) // 2
        if decoded_len(mid) < char_index:
            low = mid + 1
        else:
            high = mid
    return low
