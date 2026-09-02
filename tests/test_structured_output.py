"""OpenRouter response_format payload matches the structured-outputs docs."""
from __future__ import annotations

import unittest

from src.data_models import RAGResponse
from src.structured_output import chat_model_id, openrouter_response_format


class StructuredOutputFormatTest(unittest.TestCase):
    def test_response_format_is_strict_json_schema(self):
        payload = openrouter_response_format(RAGResponse)
        self.assertEqual(payload["type"], "json_schema")
        spec = payload["json_schema"]
        self.assertEqual(spec["name"], "RAGResponse")
        self.assertTrue(spec["strict"])
        schema = spec["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["required"],
            ["answer", "confidence", "reasoning", "sources"],
        )
        self.assertIn("answer", schema["properties"])
        self.assertNotIn("title", schema)

    def test_chat_model_id_reads_model_name_or_model(self):
        class _Named:
            model_name = "qwen/qwen3.8-flash"

        class _Model:
            model = "nvidia/nemotron-3.5-lightning:free"

        self.assertEqual(chat_model_id(_Named()), "qwen/qwen3.8-flash")
        self.assertTrue(chat_model_id(_Model()).endswith(":free"))


if __name__ == "__main__":
    unittest.main()
