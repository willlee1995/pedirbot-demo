"""Streamlit Community Cloud helpers: secrets, slim defaults, demo index."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from src.document_processor import DocumentProcessor
from src.source_allowlist import filter_live_chunks

ROOT = Path(__file__).resolve().parent.parent
USAGE_PATH = ROOT / ".demo_usage.json"

CLOUD_DEFAULTS = {
    "EMBEDDING_PROVIDER": "openrouter",
    "USE_RERANKER": "false",
    "LANGSMITH_TRACING": "false",
    "CHROMA_PERSIST_DIRECTORY": "./chroma_db",
    "COLLECTION_NAME": "pedir_demo_nemotron_embed_v6",
    "AGENT_MAX_ITERATIONS": "2",
    "TOP_K_RETRIEVAL": "4",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    "OPENROUTER_CHAT_MODEL": "qwen/qwen3.8-flash",
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


def _set_env_if_blank(key: str, value: str) -> None:
    """Write an env var only when it is missing or blank."""
    if not os.environ.get(key, "").strip():
        os.environ[key] = value


def apply_streamlit_secrets() -> None:
    """Copy Streamlit secrets into the process environment."""
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            _set_env_if_blank(str(key), str(value))
        elif hasattr(value, "items"):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)):
                    _set_env_if_blank(str(nested_key), str(nested_value))


def apply_cloud_defaults() -> None:
    """Force API embeddings, no local reranker, and no LangSmith on Cloud."""
    if not is_cloud_demo():
        return

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if has_openrouter and not has_openai:
        os.environ["LLM_PROVIDER"] = os.environ.get("LLM_PROVIDER") or "openrouter"
        os.environ["EMBEDDING_PROVIDER"] = "openrouter"
    elif has_openai and not has_openrouter:
        os.environ.setdefault("LLM_PROVIDER", "openai")
        os.environ.setdefault("EMBEDDING_PROVIDER", "openai")

    for key, value in CLOUD_DEFAULTS.items():
        os.environ.setdefault(key, value)

    # Paid CIRSE chat: do not keep a leftover :free chat secret as the default.
    chat = os.environ.get("OPENROUTER_CHAT_MODEL", "").strip()
    if not chat or chat.endswith(":free"):
        os.environ["OPENROUTER_CHAT_MODEL"] = CLOUD_DEFAULTS["OPENROUTER_CHAT_MODEL"]


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


def _demo_markdown_filenames() -> set[str]:
    """Basenames of bundled demo leaflets."""
    names: set[str] = set()
    for path in DEMO_SOURCE_DIRS:
        if path.is_dir():
            names.update(item.name for item in path.rglob("*.md"))
    return names


def _indexed_filenames(vector_store) -> set[str]:
    """Unique filenames already stored in the demo collection."""
    try:
        data = vector_store.collection.get(include=["metadatas"])
    except Exception as exc:
        logger.warning(f"Could not list demo collection metadata: {exc}")
        return set()
    names: set[str] = set()
    for meta in data.get("metadatas") or []:
        if isinstance(meta, dict) and meta.get("filename"):
            names.add(str(meta["filename"]))
    return names


def ensure_demo_knowledge_base(
    vector_store,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Ingest bundled demo markdown when the collection is empty or stale."""
    if not is_cloud_demo():
        return 0
    stats = vector_store.get_stats()
    expected = _demo_markdown_filenames()
    indexed = _indexed_filenames(vector_store)
    missing = expected - indexed
    logger.info(
        f"Demo KB collection={stats.get('collection_name')} "
        f"docs={stats.get('total_documents', 0)} "
        f"indexed_files={len(indexed)} expected_files={len(expected)} "
        f"missing={sorted(missing)}"
    )
    if stats.get("total_documents", 0) > 0 and not missing:
        return 0
    if missing and stats.get("total_documents", 0) > 0:
        logger.warning(
            f"Demo index missing {len(missing)} leaflets; "
            f"resetting {stats.get('collection_name')}"
        )
        vector_store.reset_collection()

    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        markdown_only=True,
    )
    chunks = []
    for path in DEMO_SOURCE_DIRS:
        if path.is_dir():
            chunks.extend(processor.process_directory(str(path)))
    chunks = filter_live_chunks(chunks)
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
