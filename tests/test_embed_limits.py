"""Embedding token budgets for short-context OpenRouter models."""
from __future__ import annotations

import unittest

from src.embed_limits import (
    chunk_size_for_embedding_model,
    embedding_token_limit,
    estimate_embed_tokens,
    fit_text_to_embed_limit,
)


class EmbedLimitsTest(unittest.TestCase):
    def test_liquid_has_512_token_cap(self):
        self.assertEqual(
            embedding_token_limit("liquid/lfm-2.5-embedding-350m:free"),
            512,
        )
        self.assertEqual(
            embedding_token_limit("liquid/lfm-2.5-embedding-350m🆓"),
            512,
        )
        self.assertIsNone(embedding_token_limit("nvidia/nemotron-3-embed-1b:free"))
        self.assertEqual(
            embedding_token_limit("openai/text-embedding-3-small"),
            8192,
        )

    def test_cjk_is_about_one_token_per_char(self):
        text = "腎臟活組織檢查後需要臥床觀察" * 50
        self.assertGreater(estimate_embed_tokens(text), 500)

    def test_fit_keeps_liquid_input_under_cap(self):
        text = "腎臟活組織檢查後需要臥床觀察六小時。" * 80
        fitted = fit_text_to_embed_limit(text, "liquid/lfm-2.5-embedding-350m:free")
        self.assertLessEqual(estimate_embed_tokens(fitted), 400)
        self.assertLess(len(fitted), len(text))

    def test_chunk_size_is_cjk_safe_for_liquid(self):
        size = chunk_size_for_embedding_model(
            "liquid/lfm-2.5-embedding-350m:free",
            default_chunk_size=1500,
        )
        self.assertEqual(size, 400)

    def test_openai_small_uses_8k_budget(self):
        size = chunk_size_for_embedding_model(
            "openai/text-embedding-3-small",
            default_chunk_size=1500,
        )
        self.assertEqual(size, 7500)
