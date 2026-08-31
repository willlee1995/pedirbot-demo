"""Strip model thinking tags and chain-of-thought preambles from answers."""
from __future__ import annotations

import re

_REASONING_MARKERS = (
    "here's a thinking process",
    "analyze user input",
    "evaluate context",
    "determine response strategy",
    "formulate response",
)


def _looks_like_reasoning_paragraph(paragraph: str) -> bool:
    """True for chain-of-thought outline blocks that must not reach families."""
    text = paragraph.strip()
    if not text:
        return True
    lower = text.lower()
    if lower.startswith("here's a thinking"):
        return True
    if any(marker in lower for marker in _REASONING_MARKERS):
        return True
    if lower.startswith("*(note:") or lower.startswith("(note:"):
        return True
    if re.match(r"^\d+\.\s+\*\*", text):
        return True
    return False


def strip_model_reasoning(text: str) -> str:
    """Remove model thinking tags and 'Here's a thinking process' preambles."""
    if not text:
        return text
    cleaned = re.sub(r"<unused94>thought.*?<unused95>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<unused94>", "").replace("<unused95>", "").strip()
    lower = cleaned.lower()
    if not any(marker in lower for marker in _REASONING_MARKERS):
        return cleaned
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    kept = [part for part in paragraphs if not _looks_like_reasoning_paragraph(part)]
    if kept:
        return "\n\n".join(kept).strip()
    fallback = "I don't have that information. Please ask a nurse or doctor."
    idx = lower.rfind(fallback.lower())
    if idx != -1:
        return cleaned[idx : idx + len(fallback)]
    return paragraphs[-1] if paragraphs else cleaned
