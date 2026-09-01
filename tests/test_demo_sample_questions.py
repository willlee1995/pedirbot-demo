"""Sample-question chips for the public demo must stay aligned with the guardrail."""
from __future__ import annotations

import unittest

from src.demo_sample_questions import (
    SAMPLE_QUESTIONS,
    educational_samples,
    guardrail_samples,
)

# Keep in sync with RAGPipeline.EMERGENCY_KEYWORDS / src.guardrails.EMERGENCY_KEYWORDS.
# Listed here so this test can run without installing the full LangGraph stack.
_EMERGENCY_KEYWORDS = [
    "can't breathe",
    "cannot breathe",
    "not breathing",
    "stopped breathing",
    "chest pain",
    "heart attack",
    "anaphylaxis",
    "severe allergic reaction",
    "throat swelling",
    "emergency room",
    "call 999",
    "call ambulance",
    "999",
    "unconscious",
    "passed out",
    "不能呼吸",
    "胸痛",
    "過敏反應",
    "緊急",
    "昏迷",
    "急症室",
    "uncontrolled bleeding",
    "heavy bleeding",
    "bleeding won't stop",
    "bleeding will not stop",
    "大量出血",
    "無法止血",
]


class DemoSampleQuestionsTest(unittest.TestCase):
    def test_has_educational_and_guardrail_chips(self):
        self.assertGreaterEqual(len(educational_samples()), 3)
        self.assertGreaterEqual(len(guardrail_samples()), 3)
        self.assertEqual(
            len(SAMPLE_QUESTIONS),
            len(educational_samples()) + len(guardrail_samples()),
        )

    def test_guardrail_prompts_trip_emergency_keywords(self):
        keywords = [kw.lower() for kw in _EMERGENCY_KEYWORDS]
        for sample in guardrail_samples():
            prompt_lower = sample["prompt"].lower()
            matched = any(kw in prompt_lower for kw in keywords)
            self.assertTrue(
                matched,
                f"Guardrail sample did not match emergency keywords: {sample['prompt']!r}",
            )

    def test_educational_prompts_do_not_trip_emergency_keywords(self):
        keywords = [kw.lower() for kw in _EMERGENCY_KEYWORDS]
        for sample in educational_samples():
            prompt_lower = sample["prompt"].lower()
            matched = [kw for kw in keywords if kw in prompt_lower]
            self.assertEqual(
                matched,
                [],
                f"Educational sample unexpectedly matches {matched}: {sample['prompt']!r}",
            )

    def test_labels_and_prompts_are_non_empty(self):
        for sample in SAMPLE_QUESTIONS:
            self.assertTrue(sample["label"].strip())
            self.assertTrue(sample["prompt"].strip())
            self.assertIn(sample["triggers_guardrail"], (True, False))


if __name__ == "__main__":
    unittest.main()
