"""glosa as a bench engine.

The only adapter with no extra dependency: the bench already depends on glosa,
so this is an in-process call to the same `HybridStrategy` the CLI runs.

`prepare` projects and indexes the document — no LLM call, which is the point
of §4.3's cold/warm split: glosa's cold and warm columns are the same number,
and PageIndex's are not.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from glosa.domain.hybrid import HybridConfig, HybridStrategy
from glosa.domain.index import DocIndex
from glosa.domain.values import RunStatus
from glosa.infra.docling.projection import DoclingProjector

from gbench.ports.engine import BenchDocument, Capabilities, EngineAnswer, PrepareCost

if TYPE_CHECKING:
    from glosa.ports.chat import ChatModel

    from gbench.domain.item import RenderedItem

NODE_PREFIX = "elem::"

ABSTAINING = frozenset({RunStatus.NOT_IN_DOCUMENT, RunStatus.INSUFFICIENT_EVIDENCE})
"""What counts as "this document does not answer that".

`BUDGET_EXHAUSTED` is deliberately not here: running out of budget is a failure
to read, not a finding about the document, and mapping it to `E` would pay an
engine for giving up."""


class GlosaEngine:
    def __init__(self, model: ChatModel, config: HybridConfig | None = None) -> None:
        self._model = model
        self._config = config or HybridConfig()
        self._indexes: dict[str, DocIndex] = {}

    @property
    def name(self) -> str:
        return "glosa"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            read_refs=True,
            abstention=True,
            llm_calls=True,
            groundedness=True,
            abstention_signal="status ∈ {not_in_document, insufficient_evidence}",
        )

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        started = time.monotonic()
        cached = doc.slug in self._indexes
        if not cached:
            projection = DoclingProjector().project(doc.docling_json)
            self._indexes[doc.slug] = DocIndex(projection)
        return PrepareCost(
            llm_calls=0,
            prompt_chars=0,
            wall_s=round(time.monotonic() - started, 3),
            cached=cached,
        )

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        await self.prepare(doc)
        index = self._indexes[doc.slug]
        strategy = HybridStrategy(self._model, self._config)

        started = time.monotonic()
        try:
            trace = await strategy.run(index, item.prompt)
        except Exception as exc:
            return EngineAnswer(text="", error=f"{type(exc).__name__}: {exc}")

        # Element-level self_refs, in reading order: `Step.spans` names exactly
        # what went into the excerpt, which is what `gold_refs` are authored
        # against. `Step.ref` would name the section anchor and inflate hit@k.
        read: list[str] = []
        for step in trace.steps:
            for span in step.spans:
                if span.self_ref not in read:
                    read.append(span.self_ref)

        grounded = next(
            (step.grounded for step in reversed(trace.steps) if step.grounded is not None), None
        )
        return EngineAnswer(
            text=trace.answer,
            abstained=trace.status in ABSTAINING,
            read_refs=tuple(read),
            llm_calls=trace.llm_calls,
            prompt_chars=sum(step.excerpt_chars for step in trace.steps),
            wall_s=round(time.monotonic() - started, 3),
            grounded=grounded,
            trace={
                "status": str(trace.status),
                "steps": [
                    {
                        "ref": step.ref,
                        "title": step.title,
                        "sufficient": step.sufficient,
                        "grounded": step.grounded,
                        "fallback": step.fallback,
                        "excerpt_chars": step.excerpt_chars,
                    }
                    for step in trace.steps
                ],
            },
        )

    async def aclose(self) -> None:
        await self._model.aclose()
