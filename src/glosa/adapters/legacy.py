"""Projection of a rich `Trace` onto the six-field iteration shape.

Docling Studio (and anything else built against `docling-agent`'s `RAGResult`)
consumes exactly `iteration / section_ref / reason / section_text_length /
can_answer / response`. That is a **wire** concern, not a domain one, which is
why the pydantic models live here in the adapter layer and not in
`glosa.domain.values`: the native trace can grow without ever breaking an
existing viewer, and the domain stays free of serialization types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from glosa.domain.values import RunStatus

if TYPE_CHECKING:
    from glosa.domain.values import Trace

STATUS_PREFIX = {
    RunStatus.NOT_IN_DOCUMENT: "[not answered — this document does not cover the question]",
    RunStatus.INSUFFICIENT_EVIDENCE: "[partial answer — the document did not fully answer this]",
    RunStatus.BUDGET_EXHAUSTED: "[partial answer — the run hit its budget before converging]",
}


class LegacyIteration(BaseModel):
    """Wire-compatible with `docling_agent.agent.rag_models.RAGIteration`.

    Field names and order are load-bearing: Docling Studio does
    `ReasoningIteration(**it.model_dump())`. Do not rename or add fields —
    add them to `glosa.domain.values.Step` instead.
    """

    iteration: int
    section_ref: str
    reason: str
    section_text_length: int
    can_answer: bool
    response: str


class LegacyResult(BaseModel):
    """Wire-compatible with `docling_agent.agent.rag_models.RAGResult`."""

    answer: str
    iterations: list[LegacyIteration]
    converged: bool


class IterationFactory(Protocol):
    """Anything constructible from the six legacy fields."""

    def __call__(
        self,
        *,
        iteration: int,
        section_ref: str,
        reason: str,
        section_text_length: int,
        can_answer: bool,
        response: str,
    ) -> Any: ...


class ResultFactory(Protocol):
    """Anything constructible from answer + iterations + converged."""

    def __call__(self, *, answer: str, iterations: list[Any], converged: bool) -> Any: ...


def to_legacy(
    trace: Trace,
    *,
    iteration_factory: IterationFactory = LegacyIteration,
    result_factory: ResultFactory = LegacyResult,
    annotate_status: bool = True,
) -> Any:
    """Project `trace` onto the legacy shape.

    Pass the host application's own types as factories to get them back
    directly — that is how Docling Studio avoids writing a translation adapter.

    `annotate_status` prefixes non-answered runs with a one-line marker, so a UI
    that only renders `answer` still tells the truth about what happened.
    """
    answer = trace.answer
    if annotate_status:
        prefix = STATUS_PREFIX.get(trace.status)
        if prefix:
            answer = f"{prefix}\n\n{answer}" if answer else prefix

    iterations = [
        iteration_factory(
            iteration=step.index,
            section_ref=step.ref,
            reason=step.reason,
            section_text_length=step.excerpt_chars,
            can_answer=step.sufficient,
            response=step.response,
        )
        for step in trace.steps
    ]
    return result_factory(answer=answer, iterations=iterations, converged=trace.converged)
