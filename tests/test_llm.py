"""Chat backends: request shaping, structured decoding, repair, retries."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from glosa.errors import BackendError, ReasoningParseError
from glosa.llm.ollama import OllamaChatModel
from glosa.llm.openai import OpenAIChatModel, StructuredMode
from glosa.llm.port import ChatModel, user
from glosa.llm.schema import extract_json, to_strict_schema


class Answer(BaseModel):
    ok: bool
    text: str


def _ollama(handler: Any) -> OllamaChatModel:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return OllamaChatModel(base_url="http://x", model_id="granite3.3:8b", client=client)


def _openai(handler: Any, **kwargs: Any) -> OpenAIChatModel:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return OpenAIChatModel(base_url="http://x", model_id="gpt-4.1-mini", client=client, **kwargs)


def _ollama_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def _openai_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


# -- schema helpers -----------------------------------------------------------


def test_strict_schema_forbids_extra_properties_and_requires_everything() -> None:
    schema = to_strict_schema(Answer)
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["ok", "text"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"ok": true, "text": "hi"}',
        'Sure!\n```json\n{"ok": true, "text": "hi"}\n```\nHope that helps.',
        'Here you go: {"ok": true, "text": "hi"} — done.',
    ],
)
def test_extract_json_handles_the_shapes_models_actually_emit(raw: str) -> None:
    assert json.loads(extract_json(raw) or "")["ok"] is True


def test_extract_json_ignores_braces_inside_strings() -> None:
    raw = '{"ok": true, "text": "a } brace"}'
    assert json.loads(extract_json(raw) or "")["text"] == "a } brace"


def test_extract_json_returns_none_on_prose() -> None:
    assert extract_json("I am unable to comply.") is None


# -- request shaping ----------------------------------------------------------


async def test_ollama_pushes_the_schema_into_the_decoder() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ollama_reply('{"ok": true, "text": "hi"}')

    result = await _ollama(handler).structured([user("q")], schema=Answer)
    assert result.ok is True
    assert seen["format"]["additionalProperties"] is False
    assert seen["stream"] is False
    assert seen["options"]["temperature"] == 0.0
    assert seen["options"]["seed"] == 0


async def test_openai_uses_strict_json_schema() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _openai_reply('{"ok": true, "text": "hi"}')

    await _openai(handler).structured([user("q")], schema=Answer)
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["strict"] is True


async def test_openai_degrades_to_json_object_when_asked() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _openai_reply('{"ok": false, "text": "no"}')

    model = _openai(handler, structured_mode=StructuredMode.JSON_OBJECT)
    await model.structured([user("q")], schema=Answer)
    assert seen["response_format"] == {"type": "json_object"}


# -- resilience ---------------------------------------------------------------


async def test_a_prose_reply_triggers_one_repair_round_trip() -> None:
    """This is the failure that raises `IndexError` upstream."""
    replies = ["I think it is fine.", '{"ok": true, "text": "repaired"}']

    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_reply(replies.pop(0))

    result = await _ollama(handler).structured([user("q")], schema=Answer)
    assert result.text == "repaired"
    assert not replies, "expected exactly two round-trips"


async def test_unparseable_after_repair_raises_a_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_reply("still prose")

    with pytest.raises(ReasoningParseError) as excinfo:
        await _ollama(handler).structured([user("q")], schema=Answer)
    assert excinfo.value.model_id == "granite3.3:8b"
    assert "Answer" in excinfo.value.reason


async def test_transient_5xx_is_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="warming up")
        return _ollama_reply('{"ok": true, "text": "second try"}')

    result = await _ollama(handler).structured([user("q")], schema=Answer)
    assert result.text == "second try"
    assert attempts["n"] == 2


async def test_persistent_failure_surfaces_as_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BackendError):
        await _ollama(handler).complete([user("q")])


async def test_health_is_false_when_the_host_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert await _ollama(handler).health() is False


# -- protocol conformance -----------------------------------------------------


def test_backends_satisfy_the_chat_model_port() -> None:
    assert isinstance(OllamaChatModel(model_id="m"), ChatModel)
    assert isinstance(OpenAIChatModel(model_id="m"), ChatModel)


async def test_for_model_shares_the_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_reply('{"ok": true, "text": "hi"}')

    base = _ollama(handler)
    twin = base.for_model("other:7b")
    assert twin.model_id == "other:7b"
    assert (await twin.structured([user("q")], schema=Answer)).ok is True
    await twin.aclose()
    # Closing a borrowed twin must not close the owner's client.
    assert (await base.structured([user("q")], schema=Answer)).ok is True
