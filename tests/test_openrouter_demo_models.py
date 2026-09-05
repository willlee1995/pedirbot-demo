"""Paid CIRSE default must win over leftover :free chat secrets."""
from __future__ import annotations

import unittest

from types import SimpleNamespace

from src.openrouter_demo_models import (
    DEFAULT_OPENROUTER_DEMO_MODEL_ID,
    OPENROUTER_DEMO_MODELS,
    default_demo_model_id,
    get_demo_model,
    is_paid_demo_model,
)


class OpenRouterDemoModelsTest(unittest.TestCase):
    def test_default_is_paid_gemini_flash(self):
        self.assertEqual(
            DEFAULT_OPENROUTER_DEMO_MODEL_ID,
            "google/gemini-3-flash-preview",
        )
        model = get_demo_model(DEFAULT_OPENROUTER_DEMO_MODEL_ID)
        self.assertIsNotNone(model)
        self.assertTrue(model.paid)

    def test_gemma_is_paid_medgemma_standin(self):
        model = get_demo_model("google/gemma-4-31b-it")
        self.assertIsNotNone(model)
        self.assertTrue(model.paid)
        self.assertFalse(model.id.endswith(":free"))
        self.assertIn("MedGemma", model.local_fit)
        self.assertIsNone(get_demo_model("google/gemma-4-31b-it:free"))

    def test_catalog_includes_bakeoff_pair(self):
        ids = [model.id for model in OPENROUTER_DEMO_MODELS]
        self.assertIn("google/gemini-3-flash-preview", ids)
        self.assertIn("google/gemma-4-31b-it", ids)

    def test_ignores_free_configured_secret(self):
        self.assertEqual(
            default_demo_model_id("nvidia/nemotron-3.5-lightning:free"),
            DEFAULT_OPENROUTER_DEMO_MODEL_ID,
        )

    def test_honors_paid_configured_id(self):
        self.assertEqual(
            default_demo_model_id("qwen/qwen3.8-flash"),
            "qwen/qwen3.8-flash",
        )

    def test_is_paid_works_without_paid_attribute(self):
        stale_paid = SimpleNamespace(id="google/gemini-3-flash-preview")
        stale_free = SimpleNamespace(id="nvidia/nemotron-3.5-lightning:free")
        self.assertTrue(is_paid_demo_model(stale_paid))
        self.assertFalse(is_paid_demo_model(stale_free))


if __name__ == "__main__":
    unittest.main()
