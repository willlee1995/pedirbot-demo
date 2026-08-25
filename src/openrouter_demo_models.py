"""OpenRouter free open-weight models for the Streamlit Agentic RAG demo.

Selected from the live OpenRouter :free catalog (25 Aug 2026) using
Artificial Analysis Intelligence Index v4.1.1. These are open-weight
models that can be self-hosted; the Cloud demo calls the hosted free
endpoint as a stand-in for a local hospital GPU box.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class OpenRouterDemoModel:
    """One selectable OpenRouter free model for the Streamlit test UI."""

    id: str
    short_name: str
    lab: str
    aa_index: int
    params: str
    local_fit: str
    why: str
    aa_url: str
    openrouter_url: str


# Default is the most realistic local-GPU stand-in (30B-A3B, AA 24).
DEFAULT_OPENROUTER_DEMO_MODEL_ID = "nvidia/nemotron-3.5-lightning:free"

OPENROUTER_DEMO_MODELS: List[OpenRouterDemoModel] = [
    OpenRouterDemoModel(
        id="nvidia/nemotron-3.5-lightning:free",
        short_name="Nemotron 3.5 Lightning",
        lab="NVIDIA",
        aa_index=24,
        params="30B total / 3B active (MoE)",
        local_fit="Hospital GPU box — the size you could actually run on-prem",
        why=(
            "Open-weight agent model that nearly matches Nemotron 3 Super "
            "(AA 26) at about one-quarter the size. Best default for showing "
            "that Agentic RAG still works on hardware a department could host."
        ),
        aa_url="https://artificialanalysis.ai/models/nemotron-3-5-lightning",
        openrouter_url="https://openrouter.ai/nvidia/nemotron-3.5-lightning:free",
    ),
    OpenRouterDemoModel(
        id="google/gemma-4-31b-it:free",
        short_name="Gemma 4 31B",
        lab="Google DeepMind",
        aa_index=29,
        params="30.7B dense",
        local_fit="Single workstation GPU — stronger quality, still self-hostable",
        why=(
            "Highest-quality open model in this list that still fits one "
            "workstation. Native function calling and 140+ languages, which "
            "matters for English and Cantonese patient questions."
        ),
        aa_url="https://artificialanalysis.ai/models/gemma-4-31b",
        openrouter_url="https://openrouter.ai/google/gemma-4-31b-it:free",
    ),
    OpenRouterDemoModel(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        short_name="Nemotron 3 Ultra",
        lab="NVIDIA",
        aa_index=38,
        params="550B total / 55B active (MoE)",
        local_fit="Not a typical hospital box — hosted here as the open-weight ceiling",
        why=(
            "Leading US open-weight model on the current Intelligence Index "
            "and built for agent orchestration. Too large for routine on-prem "
            "serving; use it to see the upper end of open weights on the same "
            "Agentic RAG graph."
        ),
        aa_url="https://artificialanalysis.ai/models/nvidia-nemotron-3-ultra-550b-a55b/",
        openrouter_url="https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free",
    ),
]

AA_INDEX_NOTE = (
    "Scores are Artificial Analysis Intelligence Index v4.1.1 "
    "(agents, coding, knowledge, scientific reasoning), retrieved 25 Aug 2026. "
    "Index versions are rebased over time, so treat them as a ranking, not a "
    "clinical quality score."
)


def get_demo_model(model_id: str) -> Optional[OpenRouterDemoModel]:
    """Return the catalog entry for an OpenRouter model id, if present."""
    for model in OPENROUTER_DEMO_MODELS:
        if model.id == model_id:
            return model
    return None


def default_demo_model_id(configured: str = "") -> str:
    """Prefer a configured OpenRouter id when it is in the demo catalog."""
    if configured and get_demo_model(configured):
        return configured
    return DEFAULT_OPENROUTER_DEMO_MODEL_ID


def demo_model_label(model_id: str) -> str:
    """Sidebar label: short name, local-fit class, and AA score."""
    model = get_demo_model(model_id)
    if model is None:
        return model_id
    return f"{model.short_name}  ·  AA {model.aa_index}  ·  {model.local_fit.split('—')[0].strip()}"
