"""OpenRouter structured outputs via response_format json_schema.

Docs: https://openrouter.ai/docs/guides/features/structured-outputs
LangChain with_structured_output defaults to tool/function calling, which
Qwen's Alibaba endpoint does not accept as tool_choice=required/function.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


def chat_model_id(llm: Any) -> str:
    """Return the bound chat model id from a LangChain chat model."""
    return str(
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or ""
    )


def openrouter_response_format(schema_model: Type[BaseModel]) -> Dict[str, Any]:
    """Build the OpenRouter response_format payload (strict JSON Schema)."""
    schema = schema_model.model_json_schema()
    schema.pop("title", None)
    schema["type"] = "object"
    schema["additionalProperties"] = False
    properties = schema.get("properties") or {}
    schema["required"] = list(properties)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content or "")


def invoke_openrouter_json_schema(
    llm: Any,
    messages: list,
    schema_model: Type[BaseModel],
) -> Dict[str, Any]:
    """Invoke with OpenRouter json_schema and parse into {raw, parsed, parsing_error}."""
    response_format = openrouter_response_format(schema_model)
    routed = llm
    extra_body = dict(getattr(llm, "extra_body", None) or {})
    extra_body["provider"] = {
        **(extra_body.get("provider") or {}),
        "require_parameters": True,
    }
    copier = getattr(llm, "model_copy", None)
    if callable(copier):
        try:
            routed = copier(update={"extra_body": extra_body})
        except Exception:
            routed = llm

    raw = routed.invoke(messages, response_format=response_format)
    text = _message_text(getattr(raw, "content", ""))
    parsed: Optional[BaseModel] = None
    parsing_error: Optional[Exception] = None
    if text.strip():
        try:
            parsed = schema_model.model_validate_json(text)
        except Exception as exc:
            parsing_error = exc
    else:
        parsing_error = ValueError("empty structured content")
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}
