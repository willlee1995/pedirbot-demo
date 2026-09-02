"""Tests for the public demo live source-org allowlist helpers."""
from __future__ import annotations

import unittest

from src.document_processor import DocumentChunk
from src.source_allowlist import (
    DEMO_LEAFLET_FILENAMES,
    DEMO_LEAFLET_SOURCE_ORG,
    HIDDEN_SOURCE_ORGS,
    HONG_KONG_LIVE_ORGS,
    canonicalize_live_org,
    document_is_live,
    ensure_live_ingest_metadata,
    filter_citation_sources,
    filter_live_chunks,
    filter_retrieval_hits,
    is_demo_kb_path,
    is_demo_leaflet_filename,
    source_document_title,
    unique_source_titles,
    is_hidden_source_org,
    is_live_source_org,
    live_orgs_csv,
    plan_live_search,
)
from config import LIVE_SOURCE_ORGS


class LiveSourceOrgHelpersTest(unittest.TestCase):
    def test_allowlist_is_hksir_hkch_cirse(self):
        self.assertEqual(tuple(LIVE_SOURCE_ORGS), ("HKSIR", "HKCH", "CIRSE"))
        self.assertEqual(live_orgs_csv(), "HKSIR, HKCH, CIRSE")
        self.assertEqual(DEMO_LEAFLET_SOURCE_ORG, "HKCH")
        self.assertTrue(HONG_KONG_LIVE_ORGS <= set(LIVE_SOURCE_ORGS))

    def test_canonicalize_and_membership(self):
        self.assertEqual(canonicalize_live_org("hkch"), "HKCH")
        self.assertEqual(canonicalize_live_org(" CIRSE "), "CIRSE")
        self.assertEqual(canonicalize_live_org("HKSIR"), "HKSIR")
        self.assertIsNone(canonicalize_live_org("SickKids"))
        self.assertIsNone(canonicalize_live_org("SIR"))
        self.assertIsNone(canonicalize_live_org("Unknown"))
        self.assertTrue(is_live_source_org("HKCH"))
        self.assertFalse(is_live_source_org("SickKids"))
        self.assertTrue(is_hidden_source_org("sickkids"))
        self.assertTrue(is_hidden_source_org("SIR"))
        self.assertFalse(is_hidden_source_org("HKSIR"))
        self.assertEqual(HIDDEN_SOURCE_ORGS, frozenset({"SickKids", "SIR"}))

    def test_demo_leaflets_are_hkch(self):
        self.assertEqual(len(DEMO_LEAFLET_FILENAMES), 4)
        for name in DEMO_LEAFLET_FILENAMES:
            self.assertTrue(is_demo_leaflet_filename(name))
            self.assertTrue(document_is_live({"filename": name}))
        self.assertTrue(is_demo_kb_path("demo_kb/picc_line.md"))
        self.assertTrue(is_demo_kb_path("/workspace/demo_kb/what_is_pediatric_ir.md"))
        self.assertFalse(is_demo_kb_path("KB/sickkids/picc.md"))

    def test_document_is_live_requires_allowlisted_org_otherwise(self):
        self.assertTrue(document_is_live({"source_org": "CIRSE", "filename": "x.md"}))
        self.assertFalse(document_is_live({"source_org": "SickKids", "filename": "picc.md"}))
        self.assertFalse(document_is_live({"source_org": "SIR", "filename": "guidelines.md"}))
        self.assertFalse(document_is_live({"source_org": "Unknown", "filename": "other.md"}))
        self.assertTrue(
            document_is_live({"source_org": "Unknown", "filename": "picc_line.md"})
        )

    def test_plan_live_search_rejects_hidden_org_filters(self):
        rejected = plan_live_search({"source_org": "SickKids"})
        self.assertFalse(rejected.allowed)
        self.assertIsNone(rejected.filter_dict)

        rejected_sir = plan_live_search({"source_org": "SIR"})
        self.assertFalse(rejected_sir.allowed)

        allowed = plan_live_search({"source_org": "hkch", "region": "Hong Kong"})
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.filter_dict["source_org"], "HKCH")
        self.assertEqual(allowed.filter_dict["region"], "Hong Kong")

        open_search = plan_live_search(None)
        self.assertTrue(open_search.allowed)
        self.assertIsNone(open_search.filter_dict)

        stripped = plan_live_search({"source_org": None, "region": ""})
        self.assertTrue(stripped.allowed)
        self.assertIsNone(stripped.filter_dict)

    def test_filter_retrieval_hits_drops_hidden_orgs(self):
        hits = [
            {"content": "hkch", "metadata": {"source_org": "HKCH", "filename": "a.md"}},
            {"content": "sk", "metadata": {"source_org": "SickKids", "filename": "b.md"}},
            {"content": "sir", "metadata": {"source_org": "SIR", "filename": "c.md"}},
            {"content": "cirse", "metadata": {"source_org": "CIRSE", "filename": "d.md"}},
            {"content": "leaflet", "metadata": {"filename": "fasting_and_preparation.md"}},
        ]
        kept = filter_retrieval_hits(hits)
        orgs = [item["metadata"].get("source_org") for item in kept]
        names = [item["metadata"].get("filename") for item in kept]
        self.assertEqual(len(kept), 3)
        self.assertIn("HKCH", orgs)
        self.assertIn("CIRSE", orgs)
        self.assertNotIn("SickKids", orgs)
        self.assertNotIn("SIR", orgs)
        self.assertIn("fasting_and_preparation.md", names)

    def test_filter_citation_sources_drops_hidden_orgs(self):
        sources = [
            {"source_org": "HKSIR", "filename": "local.md"},
            {"source_org": "SickKids", "filename": "sk.md"},
            {"source_org": "SIR", "filename": "society.md"},
            {"source_org": "Unknown", "filename": "mystery.md"},
            {"source_org": "HKCH", "filename": "picc_line.md"},
        ]
        kept = filter_citation_sources(sources)
        orgs = [item["source_org"] for item in kept]
        self.assertEqual(orgs, ["HKSIR", "HKCH"])

    def test_ingest_tags_demo_leaflets_and_skips_hidden(self):
        demo_meta = ensure_live_ingest_metadata({
            "source": "demo_kb/picc_line.md",
            "filename": "picc_line.md",
            "source_org": "Unknown",
        })
        self.assertIsNotNone(demo_meta)
        self.assertEqual(demo_meta["source_org"], "HKCH")
        self.assertEqual(demo_meta["region"], "Hong Kong")

        self.assertIsNone(ensure_live_ingest_metadata({
            "source": "KB/sickkids/picc.md",
            "filename": "picc.md",
            "source_org": "SickKids",
        }))
        self.assertIsNone(ensure_live_ingest_metadata({
            "source": "KB/sir/guidelines.md",
            "filename": "guidelines.md",
            "source_org": "SIR",
        }))

        cirse_meta = ensure_live_ingest_metadata({
            "source": "public/cirse_leaflet.md",
            "filename": "cirse_leaflet.md",
            "source_org": "cirse",
        })
        self.assertEqual(cirse_meta["source_org"], "CIRSE")
        self.assertNotEqual(cirse_meta.get("region"), "Hong Kong")

    def test_filter_live_chunks(self):
        chunks = [
            DocumentChunk(
                content="ok",
                metadata={"source": "demo_kb/picc_line.md", "filename": "picc_line.md"},
                chunk_id="1",
            ),
            DocumentChunk(
                content="hidden",
                metadata={"source_org": "SickKids", "filename": "sk.md"},
                chunk_id="2",
            ),
        ]
        kept = filter_live_chunks(chunks)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].metadata["source_org"], "HKCH")

    def test_source_document_title_prefers_heading_then_filename(self):
        self.assertEqual(
            source_document_title({
                "filename": "what_is_pediatric_ir.md",
                "content": "# What is paediatric interventional radiology?\n\nBody",
            }),
            "What is paediatric interventional radiology?",
        )
        self.assertEqual(
            source_document_title({"filename": "hkch_renal_biopsy_en.md"}),
            "Renal biopsy",
        )
        self.assertEqual(
            unique_source_titles([
                {"title": "PICC line (peripherally inserted central catheter)"},
                {"title": "PICC line (peripherally inserted central catheter)"},
                {"filename": "fasting_and_preparation.md"},
            ]),
            [
                "PICC line (peripherally inserted central catheter)",
                "Fasting and preparation",
            ],
        )


if __name__ == "__main__":
    unittest.main()
