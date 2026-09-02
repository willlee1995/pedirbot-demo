"""Paid CIRSE default must win over leftover :free chat secrets."""
from __future__ import annotations

import unittest

from src.openrouter_demo_models import (
    DEFAULT_OPENROUTER_DEMO_MODEL_ID,
    default_demo_model_id,
    get_demo_model,
)


class OpenRouterDemoModelsTest(unittest.TestCase):
    def test_default_is_paid_qwen_flash(self):
        self.assertEqual(DEFAULT_OPENROUTER_DEMO_MODEL_ID, "qwen/qwen3.8-flash")
        model = get_demo_model(DEFAULT_OPENROUTER_DEMO_MODEL_ID)
        self.assertIsNotNone(model)
        self.assertTrue(model.paid)

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


if __name__ == "__main__":
    unittest.main()
