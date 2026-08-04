"""Shared HTTP chat-model machinery: retries, budgets, structured decoding.

Backends differ only in how they shape a request and where the reply text sits.
Everything else — transport reuse, transient-error retry, the repair round-trip,
determinism knobs — lives here so the adapters stay a few dozen lines each.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from glosa.errors import BackendError, ReasoningParseError
from glosa.llm.port import ChatModel, Message
from glosa.llm.schema import parse_payload, repair_prompt

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class HTTPChatModel(ABC):
    """Base for HTTP chat backends.

    Args:
        base_url: Root URL of the backend.
        model_id: Model this instance talks to.
        api_key: Bearer token, when the backend wants one.
        timeout: Per-request timeout in seconds.
        temperature: Defaults to 0 — a reasoning trace nobody can reproduce is
            not much of an audit trail.
        seed: Passed through when the backend supports it, for replayable runs.
        max_retries: Retries on transient transport/5xx failures.
        client: Inject a preconfigured `httpx.AsyncClient` (tests, proxies, mTLS).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._seed = seed
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    # -- ChatModel ------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    def for_model(self, model_id: str) -> ChatModel:
        """Sibling bound to another model, sharing this instance's transport."""
        if model_id == self._model_id:
            return self
        twin = self.__class__(
            base_url=self._base_url,
            model_id=model_id,
            api_key=self._api_key,
            timeout=self._timeout,
            temperature=self._temperature,
            seed=self._seed,
            max_retries=self._max_retries,
            client=self._http,
        )
        twin._owns_client = False
        return twin

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        return await self._chat(messages, schema=None, max_tokens=max_tokens)

    async def structured[T: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: type[T],
        max_tokens: int | None = None,
    ) -> T:
        raw = await self._chat(messages, schema=schema, max_tokens=max_tokens)
        parsed = parse_payload(raw, schema)
        if parsed is not None:
            return parsed

        logger.debug("structured decode missed for %s; attempting repair", self._model_id)
        repair = [*messages, Message("assistant", raw), Message("user", repair_prompt(schema, raw))]
        repaired = await self._chat(repair, schema=schema, max_tokens=max_tokens)
        parsed = parse_payload(repaired, schema)
        if parsed is not None:
            return parsed

        raise ReasoningParseError(
            model_id=self._model_id,
            reason=f"no reply matching {schema.__name__} after constrained decoding and repair",
        )

    async def health(self) -> bool:
        try:
            response = await self._http.get(self._health_path(), headers=self._headers())
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- transport ------------------------------------------------------------

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _chat(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None,
        max_tokens: int | None,
    ) -> str:
        payload = self._build_payload(messages, schema=schema, max_tokens=max_tokens)
        path = self._chat_path()
        last: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(path, json=payload, headers=self._headers())
                if response.status_code in _RETRYABLE_STATUS:
                    raise BackendError(f"{response.status_code} from {path}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise BackendError(f"{response.status_code} from {path}: {response.text[:300]}")
                return self._extract_text(response.json())
            except (httpx.HTTPError, BackendError) as exc:
                last = exc
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(0.5 * 2**attempt)

        raise BackendError(f"{self.__class__.__name__} failed after retries: {last}") from last

    # -- backend specifics ----------------------------------------------------

    @abstractmethod
    def _chat_path(self) -> str: ...

    @abstractmethod
    def _health_path(self) -> str: ...

    @abstractmethod
    def _build_payload(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _extract_text(self, body: Any) -> str: ...
