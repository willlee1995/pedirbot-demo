"""Token budgets for OpenRouter embedding models."""
from __future__ import annotations

# Liquid LFM reports 512; leave headroom for tokenizer variance and special tokens.
_LIQUID_TOKEN_CAP = 512
_LIQUID_TOKEN_BUDGET = 400

# text-embedding-3-small / 3-large: 8192 tokens. CJK ≈ 1 token/char.
_OPENAI_EMBED_TOKEN_CAP = 8192
_OPENAI_EMBED_TOKEN_BUDGET = 7500


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
    if "text-embedding-3" in name:
        return _OPENAI_EMBED_TOKEN_CAP
    return None


def embed_token_budget(model: str) -> int | None:
    """Safe token budget under the hard cap, or None if the model is uncapped here."""
    limit = embedding_token_limit(model)
    if limit is None:
        return None
    if limit <= _LIQUID_TOKEN_CAP:
        return _LIQUID_TOKEN_BUDGET
    return _OPENAI_EMBED_TOKEN_BUDGET


def chunk_size_for_embedding_model(model: str, default_chunk_size: int) -> int:
    """Character chunk size that stays under the model token cap, including CJK."""
    budget = embed_token_budget(model)
    if budget is None:
        return default_chunk_size
    return budget


def fit_text_to_embed_limit(text: str, model: str) -> str:
    """Truncate text so an estimated token count stays under the model cap."""
    budget = embed_token_budget(model)
    if budget is None or not text:
        return text
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
