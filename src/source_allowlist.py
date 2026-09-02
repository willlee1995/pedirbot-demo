"""Live knowledge-base source-org allowlist for the public CIRSE demo.

Enforces third-party terms: only HKSIR, HKCH, and CIRSE may be retrieved,
cited, listed in the UI, advertised in prompts/tools, or ingested. Original
``demo_kb/`` leaflets are tagged HKCH so they remain searchable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from config import LIVE_SOURCE_ORGS

LIVE_SOURCE_ORG_SET = frozenset(LIVE_SOURCE_ORGS)
DEMO_LEAFLET_SOURCE_ORG = "HKCH"
HONG_KONG_LIVE_ORGS = frozenset({"HKCH", "HKSIR"})
# Named third-party orgs that must never be live sources in this public demo.
HIDDEN_SOURCE_ORGS = frozenset({"SickKids", "SIR"})
DEMO_KB_DIR_NAME = "demo_kb"
DEMO_LEAFLET_FILENAMES = frozenset({
    "picc_line.md",
    "embolization_and_aftercare.md",
    "fasting_and_preparation.md",
    "what_is_pediatric_ir.md",
})

_CANONICAL_LIVE_ORGS = {org.upper(): org for org in LIVE_SOURCE_ORGS}
_HIDDEN_ORG_KEYS = {org.upper() for org in HIDDEN_SOURCE_ORGS}


@dataclass(frozen=True)
class LiveSearchPlan:
    """How a retrieval request should apply the live-org allowlist."""

    allowed: bool
    filter_dict: Optional[Dict[str, Any]]


def live_orgs_csv() -> str:
    """Comma-separated allowlist for prompts, tool docs, and UI copy."""
    return ", ".join(LIVE_SOURCE_ORGS)


def canonicalize_live_org(value: Optional[str]) -> Optional[str]:
    """Return the canonical live org tag, or None if not allowlisted."""
    if value is None:
        return None
    key = str(value).strip().upper()
    if not key:
        return None
    return _CANONICAL_LIVE_ORGS.get(key)


def is_live_source_org(value: Optional[str]) -> bool:
    """True if ``value`` is HKSIR, HKCH, or CIRSE (case-insensitive)."""
    return canonicalize_live_org(value) is not None


def is_hidden_source_org(value: Optional[str]) -> bool:
    """True for named hidden orgs (SickKids, SIR) regardless of allowlist."""
    if value is None:
        return False
    return str(value).strip().upper() in _HIDDEN_ORG_KEYS


def is_demo_kb_path(path: Optional[str]) -> bool:
    """True if the path is under the public ``demo_kb/`` leaflet pack."""
    if not path:
        return False
    parts = {part.lower() for part in Path(str(path)).parts}
    return DEMO_KB_DIR_NAME in parts or f"/{DEMO_KB_DIR_NAME}/" in str(path).replace("\\", "/").lower()


def is_demo_leaflet_filename(filename: Optional[str]) -> bool:
    """True if the filename is one of the four original public leaflets."""
    if not filename:
        return False
    return Path(str(filename)).name.lower() in {name.lower() for name in DEMO_LEAFLET_FILENAMES}


def document_is_live(
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    source_org: Optional[str] = None,
    filename: Optional[str] = None,
) -> bool:
    """True if a document may appear in live retrieval, citations, or ingest."""
    meta: Mapping[str, Any] = metadata or {}
    org = source_org if source_org is not None else meta.get("source_org")
    if is_live_source_org(org if org is None else str(org)):
        return True
    name = filename if filename is not None else meta.get("filename")
    if is_demo_leaflet_filename(None if name is None else str(name)):
        return True
    source_path = meta.get("source")
    if is_demo_kb_path(None if source_path is None else str(source_path)):
        return True
    return False


def plan_live_search(filter_dict: Optional[Dict[str, Any]]) -> LiveSearchPlan:
    """Reject hidden ``source_org`` filters; otherwise return a safe filter dict."""
    extra = {
        key: value
        for key, value in dict(filter_dict or {}).items()
        if value is not None and str(value).strip() != ""
    }
    requested = extra.get("source_org")
    if requested is None:
        return LiveSearchPlan(allowed=True, filter_dict=extra or None)
    canonical = canonicalize_live_org(str(requested))
    if canonical is None:
        return LiveSearchPlan(allowed=False, filter_dict=None)
    extra["source_org"] = canonical
    return LiveSearchPlan(allowed=True, filter_dict=extra)


def filter_retrieval_hits(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Drop hits whose metadata is not an allowlisted live source."""
    kept: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
        top_org = result.get("source_org")
        if document_is_live(metadata, source_org=top_org if top_org is not None else None):
            kept.append(dict(result))
    return kept


def filter_citation_sources(sources: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Drop parsed citations that name a hidden or unknown source org."""
    kept: List[Dict[str, Any]] = []
    for source in sources:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
        if document_is_live(
            metadata,
            source_org=source.get("source_org"),
            filename=source.get("filename"),
        ):
            kept.append(dict(source))
    return kept


def ensure_live_ingest_metadata(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Tag original demo leaflets as HKCH. Return None to skip the document."""
    meta = dict(metadata)
    source = str(meta.get("source") or "")
    filename = str(meta.get("filename") or "")
    if is_demo_kb_path(source) or is_demo_leaflet_filename(filename):
        meta["source_org"] = DEMO_LEAFLET_SOURCE_ORG
        meta["region"] = "Hong Kong"
    org = meta.get("source_org")
    if not is_live_source_org(None if org is None else str(org)):
        return None
    canonical = canonicalize_live_org(str(org))
    if canonical:
        meta["source_org"] = canonical
    if canonical in HONG_KONG_LIVE_ORGS:
        meta["region"] = "Hong Kong"
    return meta


def heading_from_text(text: Optional[str]) -> str:
    """Return the first markdown heading in text, if any."""
    match = re.search(r"^#+\s+(.+)$", text or "", re.MULTILINE)
    return match.group(1).strip() if match else ""


def title_from_filename(filename: Optional[str]) -> str:
    """Human title from a leaflet filename."""
    stem = Path(str(filename or "")).stem.strip()
    if not stem or stem.lower() == "unknown":
        return ""
    stem = re.sub(r"^hkch_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(en|zh)$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[_-]+", " ", stem).strip().capitalize()


def source_document_title(source: Mapping[str, Any]) -> str:
    """Best leaflet title for the public source list."""
    for key in ("title", "section_title", "document_title"):
        value = str(source.get(key) or "").strip()
        if value:
            return value
    heading = heading_from_text(str(source.get("content") or ""))
    if heading:
        return heading
    pretty = title_from_filename(source.get("filename") if source.get("filename") is not None else None)
    return pretty or "Untitled leaflet"


def unique_source_titles(sources: Sequence[Mapping[str, Any]]) -> List[str]:
    """Deduplicate citation titles in retrieve order."""
    titles: List[str] = []
    seen = set()
    for source in sources:
        title = source_document_title(source)
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
    return titles


def filter_live_chunks(chunks: Iterable[Any]) -> List[Any]:
    """Keep ingest chunks that pass the live allowlist; tag demo leaflets."""
    kept: List[Any] = []
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        live_meta = ensure_live_ingest_metadata(metadata)
        if live_meta is None:
            continue
        chunk.metadata = live_meta
        kept.append(chunk)
    return kept
