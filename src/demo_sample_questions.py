"""Clickable sample questions for the public CIRSE Streamlit demo."""
from __future__ import annotations

from typing import TypedDict


class SampleQuestion(TypedDict):
    """One chip: short button label plus the full prompt that is submitted."""

    label: str
    prompt: str
    triggers_guardrail: bool


# Educational prompts map to bundled demo_kb leaflets (PICC, fasting, IR overview).
# Guardrail prompts intentionally include emergency keywords from RAGPipeline /
# EmergencyGuardrailMiddleware so presenters can show the 999 / A&E redirect.
SAMPLE_QUESTIONS: tuple[SampleQuestion, ...] = (
    {
        "label": "What is paediatric IR?",
        "prompt": "What is paediatric interventional radiology?",
        "triggers_guardrail": False,
    },
    {
        "label": "PICC: how long to fast?",
        "prompt": "How long does my child need to fast before a PICC line insertion?",
        "triggers_guardrail": False,
    },
    {
        "label": "PICC: home care tips",
        "prompt": "How should we care for a PICC line at home?",
        "triggers_guardrail": False,
    },
    {
        "label": "After embolization",
        "prompt": "What should I watch for after my child has embolization?",
        "triggers_guardrail": False,
    },
    {
        "label": "Kidney biopsy recovery",
        "prompt": "What happens after a kidney biopsy and how long does my child need to rest?",
        "triggers_guardrail": False,
    },
    {
        "label": "Guardrail: heavy bleeding",
        "prompt": (
            "My child has heavy bleeding from the puncture site and "
            "the bleeding won't stop — what should I do?"
        ),
        "triggers_guardrail": True,
    },
    {
        "label": "Guardrail: can't breathe",
        "prompt": "My child can't breathe after the procedure — please help!",
        "triggers_guardrail": True,
    },
    {
        "label": "Guardrail: chest pain",
        "prompt": "My child has chest pain and passed out after the angiogram.",
        "triggers_guardrail": True,
    },
    {
        "label": "Guardrail: 不能呼吸",
        "prompt": "我的孩子做完手術後不能呼吸，怎麼辦？",
        "triggers_guardrail": True,
    },
)


def educational_samples() -> tuple[SampleQuestion, ...]:
    """Samples that should retrieve from the leaflet pack."""
    return tuple(q for q in SAMPLE_QUESTIONS if not q["triggers_guardrail"])


def guardrail_samples() -> tuple[SampleQuestion, ...]:
    """Samples that should trip the emergency keyword screen."""
    return tuple(q for q in SAMPLE_QUESTIONS if q["triggers_guardrail"])
