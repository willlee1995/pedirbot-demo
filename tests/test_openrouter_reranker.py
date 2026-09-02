"""OpenRouter reranker reorders by API index without a live call."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.openrouter_reranker import OpenRouterReranker, uses_openrouter_rerank


class OpenRouterRerankerTest(unittest.TestCase):
    def test_slug_detection(self):
        self.assertTrue(uses_openrouter_rerank("cohere/rerank-4-fast"))
        self.assertFalse(uses_openrouter_rerank("cross-encoder/ms-marco-MiniLM-L-6-v2"))

    def test_compress_documents_follows_result_order(self):
        docs = [
            Document(page_content="first", metadata={"i": 0}),
            Document(page_content="second", metadata={"i": 1}),
            Document(page_content="third", metadata={"i": 2}),
        ]
        reranker = OpenRouterReranker(
            model="cohere/rerank-4-fast",
            api_key="test-key",
            top_n=2,
        )

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {"index": 2, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.4},
                    ]
                }

        with patch("src.openrouter_reranker.requests.post", return_value=_Resp()):
            ranked = reranker.compress_documents(docs, "query")
        self.assertEqual([doc.page_content for doc in ranked], ["third", "first"])
        self.assertEqual(ranked[0].metadata["rerank_score"], 0.9)
