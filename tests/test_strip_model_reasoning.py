"""Strip Nemotron / OpenRouter thinking preambles from family-facing answers."""
from __future__ import annotations

import unittest

from src.text_cleanup import strip_model_reasoning


class StripModelReasoningTest(unittest.TestCase):
    def test_leaves_normal_answers_alone(self):
        text = "After a kidney biopsy your child is observed for at least 6 hours."
        self.assertEqual(strip_model_reasoning(text), text)

    def test_keeps_only_the_family_facing_line(self):
        leaked = (
            "Here's a thinking process:\n\n"
            "1.  **Analyze User Input:**\n"
            "   - User asks: \"what will happen after renal biopsy\"\n\n"
            "2.  **Evaluate Context:**\n"
            "   - Document 1: Embolization and aftercare\n\n"
            "*(Note: The context provided does not contain information.)*\n\n"
            "I don't have that information. Please ask a nurse or doctor."
        )
        self.assertEqual(
            strip_model_reasoning(leaked),
            "I don't have that information. Please ask a nurse or doctor.",
        )

    def test_keeps_aftercare_after_reasoning_outline(self):
        leaked = (
            "Here's a thinking process:\n\n"
            "1.  **Analyze User Input:**\n"
            "   - User asks about renal biopsy aftercare\n\n"
            "Your child will be observed on the ward for at least 6 hours.\n\n"
            "They should lie flat in bed during that time."
        )
        cleaned = strip_model_reasoning(leaked)
        self.assertIn("observed on the ward", cleaned)
        self.assertNotIn("Here's a thinking process", cleaned)
        self.assertNotIn("Analyze User Input", cleaned)
