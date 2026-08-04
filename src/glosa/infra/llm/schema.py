"""JSON-schema plumbing and the last-resort parser.

The upstream loop asks for a ```json``` block, regex-scans the reply, and
indexes `[0]` into the result — which raises `IndexError` when the model wrote
prose instead. Studio carries an exception type and a 502 mapping just for that
failure.

Here the schema is pushed *into the decoder* (Ollama `format`, OpenAI
`json_schema`), so a malformed reply is close to unreachable. `extract_json`
exists only for backends without constrained decoding, and a failure there is a
real backend capability problem, not a formatting accident.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for `model`, tightened for strict structured-output modes.

    OpenAI's `strict: true` (and vLLM's guided decoding) require every object to
    forbid extra properties and list all of its properties as required.
    """
    schema = model.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
            properties = node.get("properties")
            if isinstance(properties, dict) and properties:
                node["required"] = list(properties)
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for value in node:
            _tighten(value)


def extract_json(text: str) -> str | None:
    """Best-effort extraction of one JSON object from a model reply.

    Tries the whole string, then fenced blocks, then the first balanced brace
    span. Returns None when nothing object-shaped is present.
    """
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fence = _from_fence(stripped)
    if fence is not None:
        return fence

    return _first_balanced_object(stripped)


def _from_fence(text: str) -> str | None:
    marker = "```"
    start = text.find(marker)
    while start != -1:
        newline = text.find("\n", start)
        if newline == -1:
            return None
        end = text.find(marker, newline)
        if end == -1:
            return None
        body = text[newline + 1 : end].strip()
        if body.startswith("{"):
            return body
        start = text.find(marker, end + len(marker))
    return None


def _first_balanced_object(text: str) -> str | None:
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_payload[T: BaseModel](text: str, schema: type[T]) -> T | None:
    """Validate a model reply against `schema`, or return None."""
    candidate = extract_json(text)
    if candidate is None:
        return None
    try:
        return schema.model_validate_json(candidate)
    except ValidationError:
        return None


def repair_prompt(schema: type[BaseModel], previous: str) -> str:
    """Instruction for a single corrective round-trip."""
    return (
        "Your previous reply could not be parsed as the required JSON object.\n\n"
        f"Previous reply:\n{previous[:1500]}\n\n"
        "Reply again with ONLY a JSON object matching this schema, no prose, no "
        "code fence:\n"
        f"{json.dumps(to_strict_schema(schema), indent=2)}"
    )
