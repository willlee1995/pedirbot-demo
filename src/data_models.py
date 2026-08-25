"""Data models for structured RAG outputs."""
from pydantic import BaseModel, Field
from typing import List, Literal

class RAGResponse(BaseModel):
    """Structured response for RAG queries."""
    answer: str = Field(description="The final answer to the user's question.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level in the answer based on the context.")
    reasoning: str = Field(description="Brief explanation of why this answer was given and how the context supports it.")
    sources: List[str] = Field(description="List of filenames or source IDs used to generate the answer.")
