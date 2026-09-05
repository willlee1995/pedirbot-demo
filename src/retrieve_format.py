"""How much retrieved text is passed into generate."""

FULL_HIT_COUNT = 3
SNIPPET_CHARS = 500


def format_retrieved_body(content: str, rank: int) -> str:
    """Full retrieved text for the top 3 hits; a short clip for the rest."""
    text = content or ""
    if rank <= FULL_HIT_COUNT or len(text) <= SNIPPET_CHARS:
        return text
    return f"{text[:SNIPPET_CHARS]}..."
