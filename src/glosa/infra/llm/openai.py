"""OpenAI-compatible backend — OpenAI, vLLM, llama.cpp server, LiteLLM, TGI…

Structured output degrades in three steps so one adapter covers the whole
family: `json_schema` (strict) → `json_object` → plain text plus the repair
round-trip inherited from `HTTPChatModel`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from glosa.domain.errors import BackendError
from glosa.infra.llm.base import HTTPChatModel
from glosa.infra.llm.schema import to_strict_schema
from glosa.ports.chat import Message

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class StructuredMode(StrEnum):
    """How much the server supports of OpenAI's structured-output API."""

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    NONE = "none"


class OpenAIChatModel(HTTPChatModel):
    """Chat model backed by any `/chat/completions` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str,
        structured_mode: StructuredMode = StructuredMode.JSON_SCHEMA,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, model_id=model_id, **kwargs)
        self._structured_mode = structured_mode

    def for_model(self, model_id: str) -> OpenAIChatModel:
        twin = super().for_model(model_id)
        assert isinstance(twin, OpenAIChatModel)
        twin._structured_mode = self._structured_mode
        return twin

    def _chat_path(self) -> str:
        return "/chat/completions"

    def _health_path(self) -> str:
        return "/models"

    def _build_payload(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "stream": False,
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens

        if schema is not None and self._structured_mode is StructuredMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": to_strict_schema(schema),
                },
            }
        elif schema is not None and self._structured_mode is StructuredMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract_text(self, body: Any) -> str:
        if not isinstance(body, dict):
            raise BackendError(f"unexpected response type: {type(body).__name__}")
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        raise BackendError(f"no content in chat completion: {str(body)[:300]}")
