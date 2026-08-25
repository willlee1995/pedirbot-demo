"""Streamlit Community Cloud helpers: secrets, slim defaults, demo index."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import List, Tuple

from src.document_processor import DocumentProcessor

ROOT = Path(__file__).resolve().parent.parent
USAGE_PATH = ROOT / ".demo_usage.json"

CLOUD_DEFAULTS = {
    "EMBEDDING_PROVIDER": "openrouter",
    "USE_RERANKER": "false",
    "LANGSMITH_TRACING": "false",
    "CHROMA_PERSIST_DIRECTORY": "./chroma_db",
    "COLLECTION_NAME": "pedir_demo_nemotron_embed",
    "AGENT_MAX_ITERATIONS": "2",
    "TOP_K_RETRIEVAL": "4",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    "OPENROUTER_CHAT_MODEL": "nvidia/nemotron-3.5-lightning:free",
    "OPENROUTER_EMBEDDING_MODEL": "nvidia/nemotron-3-embed-1b:free",
    "DEMO_ACCESS_CODE": "cirse2026",
}

DEMO_SOURCE_DIRS = (
    ROOT / "demo_kb",
)


def is_streamlit_cloud() -> bool:
    """Return True when running on Streamlit Community Cloud."""
    return (
        os.environ.get("IS_STREAMLIT_CLOUD", "").lower() == "true"
        or Path("/mount/src").exists()
    )


def is_cloud_demo() -> bool:
    """Return True for Community Cloud or an explicit local Cloud-demo run."""
    flag = os.environ.get("PEDIR_CLOUD_DEMO", "").lower()
    return is_streamlit_cloud() or flag in {"1", "true", "yes"}


def apply_streamlit_secrets() -> None:
    """Copy root-level Streamlit secrets into the process environment."""
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(str(key), str(value))


def apply_cloud_defaults() -> None:
    """Force API embeddings, no local reranker, and no LangSmith on Cloud."""
    if not is_cloud_demo():
        return

    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        os.environ.setdefault("LLM_PROVIDER", "openrouter")
        os.environ.setdefault("EMBEDDING_PROVIDER", "openrouter")
    elif os.environ.get("OPENAI_API_KEY", "").strip():
        os.environ.setdefault("LLM_PROVIDER", "openai")
        os.environ.setdefault("EMBEDDING_PROVIDER", "openai")

    for key, value in CLOUD_DEFAULTS.items():
        os.environ.setdefault(key, value)


def apply_streamlit_secrets_and_cloud_defaults() -> None:
    """Load secrets, then apply Cloud-safe defaults if this is a demo deploy."""
    apply_streamlit_secrets()
    apply_cloud_defaults()


def missing_cloud_credentials() -> List[str]:
    """Return missing API keys required for a Cloud-style demo."""
    if not is_cloud_demo():
        return []

    missing: List[str] = []
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()

    if embedding_provider == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY (embeddings)")
    needs_openrouter = embedding_provider == "openrouter" or llm_provider == "openrouter"
    if needs_openrouter and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        missing.append("OPENROUTER_API_KEY")
    if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if llm_provider not in {"openai", "openrouter", "huggingface"}:
        missing.append("LLM_PROVIDER (set openai or openrouter in Secrets)")
    return missing


def ensure_demo_knowledge_base(
    vector_store,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Ingest bundled demo markdown when the Chroma collection is empty."""
    if not is_cloud_demo():
        return 0
    stats = vector_store.get_stats()
    if stats.get("total_documents", 0) > 0:
        return 0

    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        markdown_only=True,
    )
    chunks = []
    for path in DEMO_SOURCE_DIRS:
        if path.is_dir():
            chunks.extend(processor.process_directory(str(path)))
    if not chunks:
        return 0
    vector_store.add_documents(chunks)
    return len(chunks)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def demo_access_code() -> str:
    """Shared passphrase from Secrets. Empty or 'off' means the demo is public."""
    raw = os.environ.get("DEMO_ACCESS_CODE", "").strip()
    if raw.lower() in {"off", "none", "public"}:
        return ""
    return raw


def demo_quota_limits() -> dict:
    """Return per-session, per-day, and max-character caps."""
    return {
        "session": _int_env("DEMO_MAX_QUERIES_PER_SESSION", 5),
        "day": _int_env("DEMO_MAX_QUERIES_PER_DAY", 40),
        "chars": _int_env("DEMO_MAX_QUERY_CHARS", 400),
    }


def _read_day_usage() -> Tuple[str, int]:
    today = date.today().isoformat()
    if not USAGE_PATH.exists():
        return today, 0
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return today, 0
    if data.get("date") != today:
        return today, 0
    try:
        return today, int(data.get("count", 0))
    except (TypeError, ValueError):
        return today, 0


def day_queries_used() -> int:
    """Return how many demo questions have been asked today on this instance."""
    _, count = _read_day_usage()
    return count


def record_demo_query() -> None:
    """Count one paid question against the daily file (resets at midnight)."""
    today, count = _read_day_usage()
    payload = {"date": today, "count": count + 1}
    USAGE_PATH.write_text(json.dumps(payload), encoding="utf-8")


def check_demo_query(prompt: str, session_used: int) -> str:
    """
    Return an error message if this question should not call the APIs.

    Empty string means the question may proceed.
    """
    if not is_cloud_demo():
        return ""

    limits = demo_quota_limits()
    text = (prompt or "").strip()
    if not text:
        return "Please enter a question."
    if len(text) > limits["chars"]:
        return (
            f"Please keep the question under {limits['chars']} characters "
            f"(this one is {len(text)})."
        )
    if session_used >= limits["session"]:
        return (
            f"This test session has used all {limits['session']} questions. "
            "That cap is there to save API credit. Open a new browser session "
            "only if the daily demo budget is still open."
        )
    if day_queries_used() >= limits["day"]:
        return (
            f"The shared demo has reached today's cap of {limits['day']} "
            "questions. Credits pause until tomorrow."
        )
    return ""
