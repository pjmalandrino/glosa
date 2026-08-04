"""Retrieval-first reading with a parallel frontier — the default strategy.

Two changes to how a single document gets read.

**The model stops being the search engine.** A lexical prior proposes a
shortlist in microseconds; the model spends its calls *reading* rather than
choosing. That halves the round-trips per unit (one instead of two) and removes
the outline from the critical path, so a 300-section document no longer needs
its table of contents to fit in a prompt.

**Reads happen together.** Candidates are read concurrently, so latency is
about two round-trips rather than the sum of six sequential ones. It also makes
the loop *less* greedy, not more: three plausible sections are compared on what
they actually say, instead of the first one being committed to on the strength
of its heading.

Retrieval is wrong in a predictable way — it misses paraphrase — so it never
decides alone. When the shortlist comes back empty (no lexical signal at all,
which is different from "everything scored low") the loop falls back to
`NavigateStrategy`'s model-driven hop for that round.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.budget import Budget
from glosa.domain.errors import BudgetExhausted
from glosa.domain.reading import (
    Note,
    Reading,
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
    from glosa.domain.rank import Candidate
    from glosa.ports.chat import ChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Tuning knobs. Defaults target a 30-page report on a local 8B model."""

    max_steps: int = 6
    max_llm_calls: int = 20
    deadline_s: float | None = 180.0
    outline_char_budget: int = 6_000
    excerpt_char_budget: int = 8_000
    direct_char_threshold: int = 6_000
    note_char_limit: int = 600
    max_notes: int = 8
    max_tokens: int | None = 1_024
    shortlist_size: int = 8
    """How many units the lexical prior proposes per round."""
    fanout: int = 3
    """How many of them are read concurrently, at most."""
    decisive_ratio: float = 1.5
    """When the top candidate outscores the runner-up by this much, read it
    alone. Fanout buys breadth when the shortlist is flat and wastes calls when
    it is not."""
    gap_notes: int = 2
    """How many recent findings feed the next round's query."""
    absent_votes_to_stop: int = 2


@dataclass(frozen=True, slots=True)
class _Pick:
    """A unit queued for reading, and why."""

    unit: Unit
    reason: str
    fallback: bool = False


class HybridStrategy:
    """Reads a document by retrieving first and confirming with the model."""

    def __init__(self, model: ChatModel, config: HybridConfig | None = None) -> None:
        self._model = model
        self._config = config or HybridConfig()

    async def run(self, index: DocIndex, query: str) -> Trace:
        started = time.monotonic()
        cfg = self._config
        budget = Budget(
            max_steps=cfg.max_steps,
            max_llm_calls=cfg.max_llm_calls,
            deadline_s=cfg.deadline_s,
        )

        if not index.units:
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

        return await self._loop(index, query, budget, started)

    # -- the loop -------------------------------------------------------------

    async def _loop(self, index: DocIndex, query: str, budget: Budget, started: float) -> Trace:
        cfg = self._config
        steps: list[Step] = []
        notes: list[Note] = []
        visited: list[str] = []
        absent_votes = 0
        status = RunStatus.INSUFFICIENT_EVIDENCE

        try:
            while True:
                budget.check_deadline()
                unread = [u for u in index.units if u.ref not in visited]
                if not unread:
                    break

                picks = await self._next_picks(index, query, visited, notes, budget)
                if not picks:
                    status = self._stalled_status(unread)
                    break

                results = await self._read_batch(index, query, picks, notes, budget)
                if not results:
                    status = RunStatus.BUDGET_EXHAUSTED
                    break

                answered: Reading | None = None
                for pick, reading in results:
                    steps.append(
                        make_step(
                            index=len(steps) + 1,
                            unit=pick.unit,
                            excerpt=index.excerpt(
                                pick.unit.ref,
                                char_budget=cfg.excerpt_char_budget,
                                focus=query,
                            ),
                            reason=pick.reason,
                            reading=reading,
                            fallback=pick.fallback,
                        )
                    )
                    visited.append(pick.unit.ref)
                    notes.append(
                        Note(
                            pick.unit.ref,
                            pick.unit.title,
                            reading.response[: cfg.note_char_limit],
                        )
                    )
                    if reading.sufficient and answered is None:
                        # Ties break on retrieval rank: the batch is ordered.
                        answered = reading
                    if reading.absent:
                        absent_votes += 1
                del notes[: max(0, len(notes) - cfg.max_notes)]

                if answered is not None:
                    return build_trace(
                        query=query,
                        answer=answered.response,
                        status=RunStatus.ANSWERED,
                        steps=steps,
                        budget=budget,
                        started=started,
                        model_id=self._model.model_id,
                    )

                if absent_votes >= cfg.absent_votes_to_stop:
                    status = RunStatus.NOT_IN_DOCUMENT
                    break

                if budget.steps_left == 0:
                    status = self._stalled_status([u for u in index.units if u.ref not in visited])
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

    @staticmethod
    def _stalled_status(unread: Sequence[Unit]) -> RunStatus:
        """Material left unread is a budget outcome; nothing left is an evidence one."""
        return RunStatus.BUDGET_EXHAUSTED if unread else RunStatus.INSUFFICIENT_EVIDENCE

    # -- choosing -------------------------------------------------------------

    async def _next_picks(
        self,
        index: DocIndex,
        query: str,
        visited: Sequence[str],
        notes: Sequence[Note],
        budget: Budget,
    ) -> list[_Pick]:
        """The units to read this round — retrieval first, model if it is dry."""
        cfg = self._config
        room = min(cfg.fanout, budget.steps_left, budget.calls_left)
        if room <= 0:
            return []

        probe = self._probe(index, query, notes, visited)
        candidates = (
            index.ranker.shortlist(probe, limit=cfg.shortlist_size, exclude=visited)
            if probe
            else []
        )
        if candidates:
            width = min(room, self._fanout_for(candidates))
            return [_Pick(c.unit, c.rationale) for c in candidates[:width]]

        # Either no lexical signal at all, or every query term already read.
        # Both mean the same thing: more term matching will not help. Hand the
        # round to the model, which can read intent and follow structure.
        unread = [u for u in index.units if u.ref not in visited]
        if not unread or budget.calls_left < 2:
            return []

        selection, fallback = await select_unit(
            self._model,
            query=query,
            index=index,
            candidates=unread,
            visited=visited,
            notes=notes,
            budget=budget,
            outline_char_budget=cfg.outline_char_budget,
            max_tokens=cfg.max_tokens,
        )
        unit = index.get(selection.ref)
        if unit is None:  # pragma: no cover - select_unit guarantees membership
            return []
        return [_Pick(unit, selection.reason, fallback=fallback)]

    def _probe(
        self,
        index: DocIndex,
        query: str,
        notes: Sequence[Note],
        visited: Sequence[str],
    ) -> str:
        """What to search for this round — empty means "retrieval has nothing".

        Three regimes, and they matter:

        * **first round** — search the question.
        * **question not yet exhausted** — search the question *plus* what the
          reader just said was missing. Without this the loop is "retrieve
          once, repeat" and the gap it identified is thrown away.
        * **every question term already read** — searching them again can only
          re-propose the same sections. But the reader's own words are new
          vocabulary ("look for the invoicing terms"), so search *those alone*.
          Only if there are none does retrieval stand aside for the model.
        """
        gap = " ".join(note.finding for note in notes[-self._config.gap_notes :]).strip()
        if not visited:
            return query
        if index.ranker.body.missing_terms(query, visited):
            return f"{query} {gap}".strip()
        return gap

    def _fanout_for(self, candidates: Sequence[Candidate]) -> int:
        """Breadth when the shortlist is flat, depth when it has a winner."""
        if len(candidates) < 2:
            return 1
        top, runner_up = candidates[0].score, candidates[1].score
        if runner_up > 0 and top >= self._config.decisive_ratio * runner_up:
            return 1
        return self._config.fanout

    # -- reading --------------------------------------------------------------

    async def _read_batch(
        self,
        index: DocIndex,
        query: str,
        picks: Sequence[_Pick],
        notes: Sequence[Note],
        budget: Budget,
    ) -> list[tuple[_Pick, Reading]]:
        """Read every pick concurrently; results keep the picks' order.

        Order is preserved deliberately: a trace that depends on which network
        call returned first is not reproducible, and reproducibility is the
        whole point of showing one.
        """
        cfg = self._config
        for _ in picks:
            budget.spend_step()
            budget.spend_call()

        snapshot = tuple(notes)
        tasks = [
            read_unit(
                self._model,
                query=query,
                title=pick.unit.title,
                excerpt_text=index.excerpt(
                    pick.unit.ref, char_budget=cfg.excerpt_char_budget, focus=query
                ).text,
                notes=snapshot,
                max_tokens=cfg.max_tokens,
            )
            for pick in picks
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[tuple[_Pick, Reading]] = []
        failures: list[BaseException] = []
        for pick, outcome in zip(picks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("read failed for %s: %s", pick.unit.ref, outcome)
                failures.append(outcome)
                continue
            results.append((pick, outcome))

        if not results and failures:
            # One flaky read is tolerable; a whole round failing is not, and the
            # caller needs the real error (a parse failure maps to a 502).
            raise failures[0]
        return results


__all__ = ["HybridConfig", "HybridStrategy"]
