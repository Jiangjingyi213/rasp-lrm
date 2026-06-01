from __future__ import annotations

import re


EXPLICIT_STEP_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*{1,2}|_{1,2})?\s*step\s+\d+\s*[:.)-]?\s*(?:\*{1,2}|_{1,2})?\s*"
)
BROAD_STEP_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:step\s+\d+\s*[:.)-]?|#{1,6}|\d+[\).]|[-*])\s+"
)


def segment_text(text: str, min_chars: int = 20) -> list[dict[str, int | str]]:
    text = text.strip()
    if not text:
        return []
    explicit_starts = [m.start() for m in EXPLICIT_STEP_RE.finditer(text)]
    starts = explicit_starts if len(explicit_starts) >= 2 else [m.start() for m in BROAD_STEP_RE.finditer(text)]
    if not starts:
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if p.strip()]
    else:
        starts.append(len(text))
        parts = [text[starts[i] : starts[i + 1]].strip() for i in range(len(starts) - 1)]
    segments = []
    cursor = 0
    buffer = ""
    for part in parts:
        if len(buffer) < min_chars:
            buffer = (buffer + "\n" + part).strip()
            continue
        start = text.find(buffer, cursor)
        end = start + len(buffer)
        segments.append({"segment_id": len(segments), "text": buffer, "start_char": start, "end_char": end})
        cursor = end
        buffer = part
    if buffer:
        start = text.find(buffer, cursor)
        end = start + len(buffer)
        segments.append({"segment_id": len(segments), "text": buffer, "start_char": start, "end_char": end})
    for segment in segments:
        segment["segment_type"] = classify_segment(str(segment["text"]), int(segment["segment_id"]), len(segments))
    return segments


def classify_segment(text: str, segment_id: int, num_segments: int) -> str:
    lower = text.lower()
    if "final answer" in lower or "\\boxed" in lower or segment_id == num_segments - 1 and "answer" in lower:
        return "final"
    if any(word in lower for word in ("check", "verify", "therefore", "sanity", "confirm")):
        return "verification"
    if segment_id == 0 and any(word in lower for word in ("given", "understand", "we need", "determine", "find")):
        return "understanding"
    if any(word in lower for word in ("plan", "strategy", "approach", "steps", "first")) and segment_id <= 1:
        return "planning"
    return "derivation"
