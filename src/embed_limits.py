"""Token budgets for OpenRouter embedding models."""
from __future__ import annotations

# Liquid LFM reports 512; leave headroom for tokenizer variance and special tokens.
_LIQUID_TOKEN_CAP = 512
_LIQUID_TOKEN_BUDGET = 400


def estimate_embed_tokens(text: str) -> int:
    """Conservative token count: CJK ≈ 1 token/char, other ≈ 2 chars/token."""
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
        else:
            other += 1
    return cjk + (other + 1) // 2


def embedding_token_limit(model: str) -> int | None:
    """Known hard input caps. None means use the app default chunk size."""
    name = (model or "").replace("🆓", ":free").lower()
    if "lfm" in name and "embed" in name:
        return _LIQUID_TOKEN_CAP
    return None


def chunk_size_for_embedding_model(model: str, default_chunk_size: int) -> int:
    """Character chunk size that stays under the model token cap, including CJK."""
    if embedding_token_limit(model) is None:
        return default_chunk_size
    return _LIQUID_TOKEN_BUDGET


def fit_text_to_embed_limit(text: str, model: str) -> str:
    """Truncate text so an estimated token count stays under the model cap."""
    if embedding_token_limit(model) is None or not text:
        return text
    budget = _LIQUID_TOKEN_BUDGET
    if estimate_embed_tokens(text) <= budget:
        return text
    low = 0
    high = len(text)
    best = text[:budget]
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if estimate_embed_tokens(candidate) <= budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best
