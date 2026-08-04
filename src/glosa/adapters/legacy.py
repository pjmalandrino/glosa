"""Projection of a rich `Trace` onto the six-field iteration shape.

Docling Studio (and anything else built against `docling-agent`'s `RAGResult`)
consumes exactly `iteration / section_ref / reason / section_text_length /
can_answer / response`. Keeping that projection in one small module means the
native trace can grow without ever breaking an existing viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from glosa.types import LegacyIteration, LegacyResult, RunStatus

if TYPE_CHECKING:
    from glosa.types import Trace

STATUS_PREFIX = {
    RunStatus.NOT_IN_DOCUMENT: "[not answered — this document does not cover the question]",
    RunStatus.INSUFFICIENT_EVIDENCE: "[partial answer — the document did not fully answer this]",
    RunStatus.BUDGET_EXHAUSTED: "[partial answer — the run hit its budget before converging]",
}


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
