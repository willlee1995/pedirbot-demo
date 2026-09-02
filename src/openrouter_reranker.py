"""OpenRouter rerank API (Cohere Rerank 4 Fast and similar slugs)."""
from __future__ import annotations

import os
from typing import List, Sequence

import requests
from langchain_core.documents import Document
from loguru import logger


def uses_openrouter_rerank(model: str) -> bool:
    """True for OpenRouter rerank slugs such as cohere/rerank-4-fast."""
    name = (model or "").lower()
    return name.startswith("cohere/") or "/rerank" in name


class OpenRouterReranker:
    """Reorder LangChain documents via POST /api/v1/rerank."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        top_n: int = 3,
    ):
        self.model = model
        self.top_n = top_n
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter reranker.")
        root = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.endpoint = f"{root}/rerank"
        logger.info(f"OpenRouter reranker {self.model} top_n={self.top_n}")

    def compress_documents(self, documents: Sequence[Document], query: str) -> List[Document]:
        docs = list(documents)
        if not docs:
            return []
        payload = {
            "model": self.model,
            "query": query or "",
            "documents": [doc.page_content or "" for doc in docs],
            "top_n": min(self.top_n, len(docs)),
        }
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://pedirbot-demo.streamlit.app",
                    "X-Title": "PedIR Bot",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning(f"OpenRouter rerank failed; keeping retrieve order: {exc}")
            return docs[: self.top_n]

        ranked: List[Document] = []
        for item in body.get("results") or []:
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= len(docs):
                continue
            doc = docs[index]
            score = item.get("relevance_score")
            if score is not None:
                metadata = dict(doc.metadata or {})
                metadata["rerank_score"] = score
                doc = Document(page_content=doc.page_content, metadata=metadata)
            ranked.append(doc)
        return ranked[: self.top_n] if ranked else docs[: self.top_n]
