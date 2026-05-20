from __future__ import annotations

import re


STEP_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:step\s+\d+|#{1,6}|\d+[\).]|[-*])\s+"
)


def segment_text(text: str, min_chars: int = 20) -> list[dict[str, int | str]]:
    text = text.strip()
    if not text:
        return []
    starts = [m.start() for m in STEP_RE.finditer(text)]
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
    return segments
