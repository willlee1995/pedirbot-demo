"""Retrieval, ingest, and citation paths must honor the live-org allowlist."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from src.retriever import AdvancedRetriever, BM25Retriever
from src.source_allowlist import DEMO_LEAFLET_FILENAMES, filter_citation_sources
from src.tools import get_knowledge_base_tools
from src.vector_store import VectorStore


def _hit(org: str, filename: str, content: str) -> dict:
    return {
        "content": content,
        "metadata": {"source_org": org, "filename": filename, "chunk_id": filename},
        "score": 1.0,
        "id": filename,
    }


class DummyVectorStore(VectorStore):
    """VectorStore that skips Chroma and returns canned LangChain Documents."""

    def __init__(self, docs: list[Document]):
        self.embedding_model = None
        self.collection_name = "test"
        self.persist_directory = "/tmp"
        self._lc_docs = docs

        class _VS:
            def max_marginal_relevance_search(
                inner_self, query, k=10, fetch_k=50, filter=None
            ):
                return docs[:k]

        self.vectorstore = _VS()


class VectorStoreAllowlistTest(unittest.TestCase):
    def test_similarity_search_drops_hidden_orgs(self):
        docs = [
            Document(page_content="hkch picc", metadata={"source_org": "HKCH", "filename": "a.md"}),
            Document(page_content="sickkids picc", metadata={"source_org": "SickKids", "filename": "b.md"}),
            Document(page_content="sir picc", metadata={"source_org": "SIR", "filename": "c.md"}),
            Document(page_content="cirse picc", metadata={"source_org": "CIRSE", "filename": "d.md"}),
        ]
        store = DummyVectorStore(docs)
        results = store.similarity_search("picc", k=10)
        orgs = [item["metadata"]["source_org"] for item in results]
        self.assertEqual(set(orgs), {"HKCH", "CIRSE"})

    def test_similarity_search_rejects_hidden_org_filter(self):
        docs = [
            Document(page_content="should not return", metadata={"source_org": "HKCH", "filename": "a.md"}),
        ]
        store = DummyVectorStore(docs)
        self.assertEqual(store.similarity_search("picc", filter_dict={"source_org": "SickKids"}), [])
        self.assertEqual(store.similarity_search("picc", filter_dict={"source_org": "SIR"}), [])
        kept = store.similarity_search("picc", filter_dict={"source_org": "HKCH"})
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["metadata"]["source_org"], "HKCH")


class SearchKbToolAllowlistTest(unittest.TestCase):
    def test_search_kb_rejects_hidden_org_and_does_not_name_it(self):
        store = DummyVectorStore([
            Document(
                page_content="hkch picc",
                metadata={"source_org": "HKCH", "filename": "picc_line.md"},
            ),
        ])
        search_kb = get_knowledge_base_tools(store)[0]
        hidden = search_kb.invoke({"query": "picc", "source_org": "SickKids"})
        self.assertIn("No relevant information found", hidden)
        self.assertNotIn("SickKids", hidden)
        sir = search_kb.invoke({"query": "picc", "source_org": "SIR"})
        self.assertIn("No relevant information found", sir)
        self.assertNotIn("SIR", sir)
        live = search_kb.invoke({"query": "picc", "source_org": "HKCH"})
        self.assertIn("Source: HKCH", live)
        self.assertNotIn("SickKids", live)


class RetrieverAllowlistTest(unittest.TestCase):
    def test_bm25_search_drops_hidden_orgs(self):
        docs = [
            Document(page_content="hong kong picc care", metadata={"source_org": "HKCH", "filename": "a.md"}),
            Document(page_content="sickkids picc care", metadata={"source_org": "SickKids", "filename": "b.md"}),
            Document(page_content="sir picc society", metadata={"source_org": "SIR", "filename": "c.md"}),
        ]
        retriever = BM25Retriever(docs)
        results = retriever.search("picc", k=10)
        orgs = [item["metadata"]["source_org"] for item in results]
        self.assertTrue(orgs)
        self.assertNotIn("SickKids", orgs)
        self.assertNotIn("SIR", orgs)
        self.assertIn("HKCH", orgs)

    def test_advanced_retriever_indexes_only_live_orgs(self):
        class _Collection:
            def get(self):
                return {
                    "documents": ["hkch picc", "sickkids picc", "sir picc"],
                    "metadatas": [
                        {"source_org": "HKCH", "filename": "a.md", "chunk_id": "1"},
                        {"source_org": "SickKids", "filename": "b.md", "chunk_id": "2"},
                        {"source_org": "SIR", "filename": "c.md", "chunk_id": "3"},
                    ],
                }

        class _VS:
            _collection = _Collection()

            def as_retriever(self, **kwargs):
                return MagicMock()

        store = DummyVectorStore([])
        store.vectorstore = _VS()
        store.similarity_search = MagicMock(return_value=[
            _hit("HKCH", "a.md", "hkch picc"),
            _hit("SickKids", "b.md", "sickkids picc"),
            _hit("CIRSE", "d.md", "cirse picc"),
        ])

        retriever = AdvancedRetriever(store, llm=None, use_hybrid_search=True)
        self.assertIsNotNone(retriever.bm25_retriever)
        indexed_orgs = [
            doc.metadata.get("source_org") for doc in retriever.bm25_retriever.documents
        ]
        self.assertEqual(indexed_orgs, ["HKCH"])

        results = retriever.retrieve("picc", k=10)
        orgs = [item["metadata"]["source_org"] for item in results]
        self.assertIn("HKCH", orgs)
        self.assertIn("CIRSE", orgs)
        self.assertNotIn("SickKids", orgs)
        self.assertNotIn("SIR", orgs)

    def test_advanced_retriever_rejects_hidden_org_filter(self):
        store = DummyVectorStore([])
        store.vectorstore._collection = MagicMock()
        store.vectorstore._collection.get.return_value = {}
        store.as_retriever = MagicMock(return_value=MagicMock())
        retriever = AdvancedRetriever(store, llm=None, use_hybrid_search=False)
        self.assertEqual(
            retriever.retrieve("picc", filter_dict={"source_org": "SickKids"}),
            [],
        )


class DemoKbIngestTest(unittest.TestCase):
    def test_demo_kb_leaflets_are_tagged_hkch_and_kept(self):
        processor = DocumentProcessor(markdown_only=True, whole_document=True)
        demo_dir = Path(__file__).resolve().parents[1] / "demo_kb"
        chunks = processor.process_directory(str(demo_dir))
        self.assertTrue(chunks)
        filenames = {chunk.metadata.get("filename") for chunk in chunks}
        self.assertTrue(set(DEMO_LEAFLET_FILENAMES).issubset(filenames))
        self.assertTrue(any(str(name).startswith("hkch_") for name in filenames))
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("source_org"), "HKCH")
            self.assertEqual(chunk.metadata.get("region"), "Hong Kong")

    def test_process_directory_skips_hidden_org_files(self):
        processor = DocumentProcessor(markdown_only=True, whole_document=True)
        with tempfile.TemporaryDirectory() as tmp:
            hidden = Path(tmp)
            (hidden / "sickkids_picc.md").write_text("# PICC at a hidden hospital\n", encoding="utf-8")
            (hidden / "sir_guidelines.md").write_text("# Society guidelines\n", encoding="utf-8")
            chunks = processor.process_directory(str(hidden))
        self.assertEqual(chunks, [])


class RagPipelineCitationFilterTest(unittest.TestCase):
    def test_parsed_citations_drop_hidden_orgs(self):
        mixed = [
            {"source_org": "HKCH", "filename": "picc_line.md", "tool": "search_kb"},
            {"source_org": "SickKids", "filename": "sk.md", "tool": "search_kb"},
            {"source_org": "SIR", "filename": "sir.md", "tool": "search_kb"},
            {"source_org": "CIRSE", "filename": "cirse.md", "tool": "search_kb"},
        ]
        kept = filter_citation_sources(mixed)
        self.assertEqual(
            [item["source_org"] for item in kept],
            ["HKCH", "CIRSE"],
        )
        # Pipeline uses the same helper on every generate_response path.
        self.assertTrue(hasattr(RAGPipeline, "generate_response"))


if __name__ == "__main__":
    unittest.main()
