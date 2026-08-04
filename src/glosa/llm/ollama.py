"""Ollama backend.

Uses `/api/chat` with `format` set to the JSON schema — Ollama constrains
decoding to it, so the reply is object-shaped by construction. No `mellea`, no
`ModelIdentifier` wrapping, no process-wide `OLLAMA_HOST` mutation: the host is
an argument.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from glosa.errors import BackendError
from glosa.llm.base import HTTPChatModel
from glosa.llm.port import Message
from glosa.llm.schema import to_strict_schema

DEFAULT_HOST = "http://localhost:11434"


class OllamaChatModel(HTTPChatModel):
    """Chat model backed by an Ollama server."""

    def __init__(self, *, base_url: str = DEFAULT_HOST, model_id: str, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, model_id=model_id, **kwargs)

    def _chat_path(self) -> str:
        return "/api/chat"

    def _health_path(self) -> str:
        return "/api/tags"

    def _build_payload(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self._temperature}
        if self._seed is not None:
            options["seed"] = self._seed
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if schema is not None:
            payload["format"] = to_strict_schema(schema)
        return payload

    def _extract_text(self, body: Any) -> str:
        if not isinstance(body, dict):
            raise BackendError(f"unexpected Ollama response type: {type(body).__name__}")
        message = body.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        fallback = body.get("response")  # /api/generate shape, just in case
        if isinstance(fallback, str):
            return fallback
        raise BackendError(f"no content in Ollama response: {str(body)[:300]}")
