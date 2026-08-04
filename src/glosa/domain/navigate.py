"""Outline-guided reading: the model chooses where to look, one hop at a time.

This is the shape of the upstream chunkless loop — select, read, decide — with
the parts that made it fragile replaced (flat notes memory instead of a growing
session, four honest outcomes instead of one boolean, revisiting allowed,
recoverable selection, hard budgets).

It is kept as its own strategy because it is the right answer when lexical
retrieval has nothing to say: a question phrased entirely differently from the
document's vocabulary. `HybridStrategy` uses it as its fallback rather than
reimplementing it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.budget import Budget
from glosa.domain.errors import BudgetExhausted
from glosa.domain.reading import (
    Note,
    Reading,
    Selection,
    build_trace,
    compose,
    make_step,
    read_unit,
    read_whole_document,
    select_unit,
)
from glosa.domain.values import RunStatus, Step, Trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.index import DocIndex, Unit
    from glosa.ports.chat import ChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NavigateConfig:
    """Tuning knobs. Defaults target a 30-page report on a local 8B model."""

    max_steps: int = 6
    max_llm_calls: int = 20
    deadline_s: float | None = 180.0
    outline_char_budget: int = 6_000
    excerpt_char_budget: int = 8_000
    direct_char_threshold: int = 6_000
    note_char_limit: int = 600
    max_notes: int = 8
    allow_revisit: bool = True
    max_tokens: int | None = 1_024
    absent_votes_to_stop: int = 2


class NavigateStrategy:
    """Reads a document by walking its outline."""

    def __init__(self, model: ChatModel, config: NavigateConfig | None = None) -> None:
        self._model = model
        self._config = config or NavigateConfig()

    async def run(self, index: DocIndex, query: str) -> Trace:
        started = time.monotonic()
        cfg = self._config
        budget = Budget(
            max_steps=cfg.max_steps,
            max_llm_calls=cfg.max_llm_calls,
            deadline_s=cfg.deadline_s,
        )
        units = index.units

        if not units:
            return build_trace(
                query=query,
                answer="The document contains no readable content.",
                status=RunStatus.NOT_IN_DOCUMENT,
                steps=(),
                budget=budget,
                started=started,
                model_id=self._model.model_id,
            )

        if index.total_chars <= cfg.direct_char_threshold:
            step, status = await read_whole_document(
                self._model,
                index=index,
                query=query,
                budget=budget,
                excerpt_char_budget=cfg.excerpt_char_budget,
                max_tokens=cfg.max_tokens,
            )
            return build_trace(
                query=query,
                answer=step.response,
                status=status,
                steps=(step,),
                budget=budget,
                started=started,
                model_id=self._model.model_id,
            )

        return await self._loop(index, units, query, budget, started)

    async def _loop(
        self,
        index: DocIndex,
        units: Sequence[Unit],
        query: str,
        budget: Budget,
        started: float,
    ) -> Trace:
        cfg = self._config
        steps: list[Step] = []
        notes: list[Note] = []
        visited: list[str] = []
        absent_votes = 0
        status = RunStatus.INSUFFICIENT_EVIDENCE

        try:
            while True:
                budget.check_deadline()
                candidates = [u for u in units if u.ref not in visited]
                revisiting = False
                if not candidates:
                    if not cfg.allow_revisit:
                        break
                    candidates = list(units)
                    revisiting = True

                budget.spend_step()
                selection, fallback = await select_unit(
                    self._model,
                    query=query,
                    units=units,
                    candidates=candidates,
                    visited=visited,
                    notes=notes,
                    budget=budget,
                    outline_char_budget=cfg.outline_char_budget,
                    max_tokens=cfg.max_tokens,
                )
                unit = index.get(selection.ref)
                if unit is None:  # pragma: no cover - select_unit guarantees membership
                    break

                excerpt = index.excerpt(unit.ref, char_budget=cfg.excerpt_char_budget, focus=query)
                budget.spend_call()
                reading = await read_unit(
                    self._model,
                    query=query,
                    title=unit.title,
                    excerpt_text=excerpt.text,
                    notes=tuple(notes),
                    max_tokens=cfg.max_tokens,
                )

                steps.append(
                    make_step(
                        index=len(steps) + 1,
                        unit=unit,
                        excerpt=excerpt,
                        reason=selection.reason,
                        reading=reading,
                        revisited=revisiting or unit.ref in visited,
                        fallback=fallback,
                    )
                )
                visited.append(unit.ref)
                notes.append(Note(unit.ref, unit.title, reading.response[: cfg.note_char_limit]))
                del notes[: max(0, len(notes) - cfg.max_notes)]

                if reading.sufficient:
                    return build_trace(
                        query=query,
                        answer=reading.response,
                        status=RunStatus.ANSWERED,
                        steps=steps,
                        budget=budget,
                        started=started,
                        model_id=self._model.model_id,
                    )

                if reading.absent:
                    absent_votes += 1
                    if absent_votes >= cfg.absent_votes_to_stop:
                        status = RunStatus.NOT_IN_DOCUMENT
                        break

                if budget.steps >= cfg.max_steps:
                    # Stopping with material left unread is a budget outcome;
                    # stopping having read everything is an evidence outcome.
                    unread = [u for u in units if u.ref not in visited]
                    status = (
                        RunStatus.BUDGET_EXHAUSTED if unread else RunStatus.INSUFFICIENT_EVIDENCE
                    )
                    break
        except BudgetExhausted as exc:
            logger.info("run stopped early: %s", exc)
            status = RunStatus.BUDGET_EXHAUSTED

        answer = await compose(self._model, query=query, notes=notes, budget=budget)
        return build_trace(
            query=query,
            answer=answer,
            status=status,
            steps=steps,
            budget=budget,
            started=started,
            model_id=self._model.model_id,
            notes=notes,
        )


__all__ = ["NavigateConfig", "NavigateStrategy", "Note", "Reading", "Selection"]
