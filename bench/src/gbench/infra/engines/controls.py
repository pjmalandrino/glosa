"""The two engines that do not navigate.

They implement the same port as the real ones, which is the point: the controls
go through the same rendering, the same letter parser and the same scorer, so a
number from them is comparable to a number from an engine by construction
rather than by argument.

**closed-book** — the model, the item, no document. Its accuracy is the floor
below which a reading engine has demonstrated nothing. Items it answers are
flagged `leaky` and drop out of the headline.

**oracle-context** — the model, the item, and only the text of `gold_refs`. Its
accuracy is the ceiling this model can reach with perfect retrieval, so
`oracle` minus `engine` is retrieval headroom and `100` minus `oracle` is the model's own
limit, which no reading loop can fix. Without it, a 62 % score is unreadable:
it could be a bad navigator or a model that cannot do the arithmetic.

The oracle is *told* the gold refs. That is not a leak, it is the definition of
a ceiling — and it never sees `item.answer`, which the port forbids.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from glosa.ports.chat import system, user

from gbench.ports.engine import BenchDocument, Capabilities, EngineAnswer, PrepareCost

if TYPE_CHECKING:
    from collections.abc import Callable

    from glosa.ports.chat import ChatModel

    from gbench.domain.item import RenderedItem

_SYSTEM = "You answer multiple-choice questions. Reply with a single letter and nothing else."

_CLOSED_BOOK_NOTE = (
    "You do not have the document. Answer from what you know; if you cannot, answer E."
)


class ClosedBookEngine:
    """The floor: no document at all."""

    def __init__(self, model: ChatModel, *, max_tokens: int = 8) -> None:
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "closed-book"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(llm_calls=True, abstention_signal="none (letter E only)")

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        return PrepareCost(llm_calls=0, prompt_chars=0)

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        prompt = f"{_CLOSED_BOOK_NOTE}\n\n{item.prompt}"
        started = time.monotonic()
        try:
            text = await self._model.complete(
                [system(_SYSTEM), user(prompt)], max_tokens=self._max_tokens
            )
        except Exception as exc:
            return EngineAnswer(text="", error=f"{type(exc).__name__}: {exc}")
        return EngineAnswer(
            text=text,
            llm_calls=1,
            prompt_chars=len(prompt),
            wall_s=round(time.monotonic() - started, 3),
        )

    async def aclose(self) -> None:
        await self._model.aclose()


class OracleContextEngine:
    """The ceiling: perfect retrieval, no navigation."""

    def __init__(
        self,
        model: ChatModel,
        text_of: Callable[[str, str], str],
        *,
        max_tokens: int = 8,
        char_budget: int = 8_000,
    ) -> None:
        self._model = model
        self._text_of = text_of
        self._max_tokens = max_tokens
        self._char_budget = char_budget

    @property
    def name(self) -> str:
        return "oracle-context"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            read_refs=True, llm_calls=True, abstention_signal="none (letter E only)"
        )

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        return PrepareCost(llm_calls=0, prompt_chars=0)

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        refs = item.item.gold_refs
        context = "\n\n".join(self._text_of(doc.slug, ref) for ref in refs)[: self._char_budget]
        # A `not_stated` item has no gold ref by construction: the ceiling for
        # "is this absent?" is the model reading the section it was removed
        # from and finding nothing.
        if not context and item.item.removed_from:
            context = self._text_of(doc.slug, item.item.removed_from)[: self._char_budget]
        prompt = f"Document extract:\n\n{context}\n\n---\n\n{item.prompt}"

        started = time.monotonic()
        try:
            text = await self._model.complete(
                [system(_SYSTEM), user(prompt)], max_tokens=self._max_tokens
            )
        except Exception as exc:
            return EngineAnswer(text="", error=f"{type(exc).__name__}: {exc}")
        return EngineAnswer(
            text=text,
            read_refs=refs,
            llm_calls=1,
            prompt_chars=len(prompt),
            wall_s=round(time.monotonic() - started, 3),
        )

    async def aclose(self) -> None:
        await self._model.aclose()
