"""Post-procedure caution keywords for the HIGH warning suffix.

Kept as a tiny import so RAGPipeline can append the suffix even if a cached
SafetyGuard instance predates the caution path.
"""
from __future__ import annotations

# Infection / wound concerns: answer from leaflets, then append HIGH warning.
# Not full emergencies (those stay on EMERGENCY_KEYWORDS).
CAUTION_KEYWORDS: tuple[str, ...] = (
    "oozing",
    "pus",
    "foul smell",
    "very red",
    "looks red",
    "spreading redness",
    "red streaks",
    "yellow or green",
    "流膿",
    "發紅",
    "紅腫擴散",
)


def caution_keyword_hits(text: str) -> list[str]:
    """Return caution keywords found in text (case-insensitive)."""
    text_lower = (text or "").lower()
    return [kw for kw in CAUTION_KEYWORDS if kw.lower() in text_lower]
