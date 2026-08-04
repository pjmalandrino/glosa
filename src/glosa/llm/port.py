"""The only seam between glosa's reasoning and an LLM vendor.

Nothing under `document/`, `strategy/` or `adapters/` imports an SDK or an HTTP
client; everything crosses here. That is what makes the backend a configuration
choice instead of a rewrite — the constraint Docling Studio currently documents
as "today only OLLAMA is realizable".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


@runtime_checkable
class ChatModel(Protocol):
    """A chat backend that can be asked for free text or for a typed object."""

    @property
    def model_id(self) -> str:
        """Identifier of the model this instance talks to."""
        ...

    def for_model(self, model_id: str) -> ChatModel:
        """Return a sibling bound to another model, sharing this one's transport.

        Used for the per-request `model_id` override in Studio's port.
        """
        ...

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Free-form completion."""
        ...

    async def structured[T: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: type[T],
        max_tokens: int | None = None,
    ) -> T:
        """Completion constrained to `schema`, validated before it is returned.

        Implementations must use the backend's native constrained decoding when
        available and fall back to a repair round-trip otherwise. They raise
        `ReasoningParseError` only once both have failed.
        """
        ...

    async def health(self) -> bool:
        """Cheap reachability probe. Must not raise."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...
