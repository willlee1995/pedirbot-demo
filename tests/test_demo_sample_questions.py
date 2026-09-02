"""Sample-question chips for the public demo must stay aligned with the guardrail."""
from __future__ import annotations

import unittest

from pathlib import Path

from src.caution_keywords import CAUTION_KEYWORDS
from src.demo_sample_questions import (
    SAMPLE_QUESTIONS,
    caution_samples,
    educational_samples,
    guardrail_samples,
)

# Keep in sync with RAGPipeline.EMERGENCY_KEYWORDS / src.guardrails.EMERGENCY_KEYWORDS.
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

_CAUTION_KEYWORDS = list(CAUTION_KEYWORDS)


class DemoSampleQuestionsTest(unittest.TestCase):
    def test_has_educational_guardrail_and_caution_chips(self):
        self.assertGreaterEqual(len(educational_samples()), 3)
        self.assertGreaterEqual(len(guardrail_samples()), 3)
        self.assertGreaterEqual(len(caution_samples()), 1)
        self.assertEqual(
            len(SAMPLE_QUESTIONS),
            len(educational_samples())
            + len(guardrail_samples())
            + len(caution_samples()),
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

    def test_educational_prompts_do_not_trip_emergency_or_caution(self):
        emergency = [kw.lower() for kw in _EMERGENCY_KEYWORDS]
        caution = [kw.lower() for kw in _CAUTION_KEYWORDS]
        for sample in educational_samples():
            prompt_lower = sample["prompt"].lower()
            matched_e = [kw for kw in emergency if kw in prompt_lower]
            matched_c = [kw for kw in caution if kw in prompt_lower]
            self.assertEqual(
                matched_e,
                [],
                f"Educational sample unexpectedly matches emergency {matched_e}: {sample['prompt']!r}",
            )
            self.assertEqual(
                matched_c,
                [],
                f"Educational sample unexpectedly matches caution {matched_c}: {sample['prompt']!r}",
            )

    def test_caution_prompts_hit_caution_keywords_not_emergency(self):
        emergency = [kw.lower() for kw in _EMERGENCY_KEYWORDS]
        caution = [kw.lower() for kw in _CAUTION_KEYWORDS]
        for sample in caution_samples():
            prompt_lower = sample["prompt"].lower()
            matched_c = [kw for kw in caution if kw in prompt_lower]
            matched_e = [kw for kw in emergency if kw in prompt_lower]
            self.assertTrue(
                matched_c,
                f"Caution sample did not match caution keywords: {sample['prompt']!r}",
            )
            self.assertEqual(
                matched_e,
                [],
                f"Caution sample must not trip emergency keywords {matched_e}: {sample['prompt']!r}",
            )

    def test_high_warning_suffix_is_hkch_ir_nurse_contact(self):
        root = Path(__file__).resolve().parents[1]
        safety = (root / "src" / "safety_guard.py").read_text(encoding="utf-8")
        caption = (root / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("HKCH IR nurse contact", safety)
        self.assertIn("3513 6099", safety)
        self.assertIn("HKCH IR nurse contact, 3513 6099", caption)

    def test_labels_and_prompts_are_non_empty(self):
        for sample in SAMPLE_QUESTIONS:
            self.assertTrue(sample["label"].strip())
            self.assertTrue(sample["prompt"].strip())
            self.assertIn(sample["triggers_guardrail"], (True, False))
            self.assertIn(sample["triggers_caution"], (True, False))
            self.assertFalse(
                sample["triggers_guardrail"] and sample["triggers_caution"],
                f"Sample cannot be both guardrail and caution: {sample['label']}",
            )


if __name__ == "__main__":
    unittest.main()
