"""OpenRouter chat models for the Streamlit Agentic RAG demo.

Default is paid Gemini 3 Flash Preview — the Eval 1 bake-off model.
Gemma 4 31B (paid, not :free) stands in for MedGemma 1.5 (no hosted API).
Free open-weight slugs stay in the picker as a hospital-GPU stand-in.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class OpenRouterDemoModel:
    """One selectable OpenRouter model for the Streamlit test UI."""

    id: str
    short_name: str
    lab: str
    aa_index: int
    params: str
    local_fit: str
    why: str
    aa_url: str
    openrouter_url: str
    paid: bool = False


# Paid CIRSE demo default — the actual Eval 1 bake-off model.
DEFAULT_OPENROUTER_DEMO_MODEL_ID = "google/gemini-3-flash-preview"

OPENROUTER_DEMO_MODELS: List[OpenRouterDemoModel] = [
    OpenRouterDemoModel(
        id="google/gemini-3-flash-preview",
        short_name="Gemini 3 Flash",
        lab="Google",
        aa_index=0,
        params="Flash Preview (paid OpenRouter)",
        local_fit="Eval 1 bake-off model — the hosted model we scored",
        why=(
            "Same Gemini 3 Flash Preview used in the two-rater bake-off "
            "(Eval 1). Paid OpenRouter route so generate does not sit in "
            "the :free queue."
        ),
        aa_url="https://openrouter.ai/google/gemini-3-flash-preview",
        openrouter_url="https://openrouter.ai/google/gemini-3-flash-preview",
        paid=True,
    ),
    OpenRouterDemoModel(
        id="google/gemma-4-31b-it",
        short_name="Gemma 4 31B",
        lab="Google DeepMind",
        aa_index=29,
        params="30.7B dense (paid OpenRouter)",
        local_fit="MedGemma 1.5 stand-in — no hosted MedGemma API",
        why=(
            "Eval 2 used MedGemma 1.5 on-prem. There is no hosted MedGemma "
            "1.5 API, so this paid Gemma 4 31B slug is the closest Google "
            "open-weight stand-in with a reliable OpenRouter queue (not the "
            ":free variant)."
        ),
        aa_url="https://artificialanalysis.ai/models/gemma-4-31b",
        openrouter_url="https://openrouter.ai/google/gemma-4-31b-it",
        paid=True,
    ),
    OpenRouterDemoModel(
        id="qwen/qwen3.8-flash",
        short_name="Qwen 3.8 Flash",
        lab="Alibaba / Qwen",
        aa_index=0,
        params="Flash (paid OpenRouter)",
        local_fit="Paid low-latency alternative",
        why=(
            "Paid OpenRouter route if you want a cheaper/faster generate "
            "than Gemini on the same Agentic RAG graph. Embeddings stay on "
            "the free OpenRouter embed slug."
        ),
        aa_url="https://openrouter.ai/qwen/qwen3.8-flash",
        openrouter_url="https://openrouter.ai/qwen/qwen3.8-flash",
        paid=True,
    ),
    OpenRouterDemoModel(
        id="nvidia/nemotron-3.5-lightning:free",
        short_name="Nemotron 3.5 Lightning",
        lab="NVIDIA",
        aa_index=24,
        params="30B total / 3B active (MoE)",
        local_fit="Hospital GPU box — the size you could actually run on-prem",
        why=(
            "Open-weight agent model that nearly matches Nemotron 3 Super "
            "(AA 26) at about one-quarter the size. Best free stand-in for "
            "hardware a department could host."
        ),
        aa_url="https://artificialanalysis.ai/models/nemotron-3-5-lightning",
        openrouter_url="https://openrouter.ai/nvidia/nemotron-3.5-lightning:free",
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
    "Free-model scores are Artificial Analysis Intelligence Index v4.1.1 "
    "(agents, coding, knowledge, scientific reasoning), retrieved 25 Aug 2026. "
    "Index versions are rebased over time, so treat them as a ranking, not a "
    "clinical quality score. Gemini 3 Flash, Gemma 4 31B, and Qwen 3.8 Flash "
    "are paid OpenRouter routes and are not ranked on that free-model list."
)


def is_paid_demo_model(model: object) -> bool:
    """True for paid catalog rows. Works if a stale Cloud class lacks `.paid`."""
    flagged = getattr(model, "paid", None)
    if flagged is not None:
        return bool(flagged)
    model_id = getattr(model, "id", "") or ""
    return bool(model_id) and not str(model_id).endswith(":free")


def get_demo_model(model_id: str) -> Optional[OpenRouterDemoModel]:
    """Return the catalog entry for an OpenRouter model id, if present."""
    for model in OPENROUTER_DEMO_MODELS:
        if model.id == model_id:
            return model
    return None


def default_demo_model_id(configured: str = "") -> str:
    """Prefer the paid CIRSE default; ignore leftover :free chat secrets."""
    if (
        configured
        and not configured.endswith(":free")
        and get_demo_model(configured)
    ):
        return configured
    return DEFAULT_OPENROUTER_DEMO_MODEL_ID


def demo_model_label(model_id: str) -> str:
    """Sidebar label: short name, paid/AA class, and local-fit class."""
    model = get_demo_model(model_id)
    if model is None:
        return model_id
    fit = model.local_fit.split("—")[0].strip()
    if is_paid_demo_model(model):
        return f"{model.short_name}  ·  paid  ·  {fit}"
    return f"{model.short_name}  ·  AA {model.aa_index}  ·  {fit}"
