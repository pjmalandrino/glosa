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
decides alone. When the shortlist is not worth believing, the model is asked to
pick, and both picks are read in the same round.

This is the only reading loop in the package. An earlier version kept a second,
purely model-driven strategy alongside it as a fallback; the hedge subsumed it,
and two loops that can drift into producing different traces for the same
document is not a feature. The baseline it represented lives in
`tests/baseline.py`, where it belongs: something to measure against, not to
ship.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.budget import Budget
from glosa.domain.errors import BudgetExhausted, ReasoningParseError
from glosa.domain.rank import Shortlist
from glosa.domain.reading import (
    Note,
    Reading,
    build_trace,
    compose,
    expand_query,
    make_step,
    read_unit,
    read_whole_document,
    select_unit,
)
from glosa.domain.values import RunStatus, Step, Trace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from glosa.domain.index import DocIndex, Unit
    from glosa.ports.chat import ChatModel

logger = logging.getLogger(__name__)

NOTHING_FOUND = "No relevant content was found in this document."


async def _within_deadline[T](budget: Budget, pending: Awaitable[T]) -> T:
    """Run one awaitable under what is left of the wall-clock budget.

    The deadline used to be checked only between rounds, so one hung backend
    call could overshoot it by minutes. Bounding every call is what turns
    `deadline_s` from a hint into a contract: at the deadline, in-flight reads
    are cancelled and the run ends as `BUDGET_EXHAUSTED` with whatever it has.
    """
    remaining = budget.remaining_s
    if remaining is None:
        return await pending
    try:
        async with asyncio.timeout(max(remaining, 0.0)):
            return await pending
    except TimeoutError as exc:
        raise BudgetExhausted(
            f"deadline of {budget.deadline_s or 0:.0f}s reached mid-call"
        ) from exc


HEDGE_WIDTH = 2
"""Slots a hedged round needs: the lexical guess *and* the model's own pick.

Reading only one of them is not a hedge, it is the guess it was supposed to
check."""


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
    min_coverage: float = 0.34
    """Share of the question's answerable terms the top candidate must contain."""
    min_margin: float = 1.15
    """How much the leader must beat the runner-up to be believed. Around 1.0
    the ranking is flat — retrieval is guessing, and saying so is the point."""
    expand_query: bool = True
    """Let the model write retrieval vocabulary when the ranking is unsure."""
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
            try:
                step, status = await _within_deadline(
                    budget,
                    read_whole_document(
                        self._model,
                        index=index,
                        query=query,
                        budget=budget,
                        excerpt_char_budget=cfg.excerpt_char_budget,
                        max_tokens=cfg.max_tokens,
                    ),
                )
            except BudgetExhausted as exc:
                # Same contract as the loop path: running out is an outcome,
                # not a crash.
                logger.info("run stopped early: %s", exc)
                return build_trace(
                    query=query,
                    answer="",
                    status=RunStatus.BUDGET_EXHAUSTED,
                    steps=(),
                    budget=budget,
                    started=started,
                    model_id=self._model.model_id,
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
        expansion: tuple[str, ...] | None = None
        status = RunStatus.INSUFFICIENT_EVIDENCE

        try:
            while True:
                budget.check_deadline()
                unread = [u for u in index.units if u.ref not in visited]
                if not unread:
                    break

                picks, expansion = await self._next_picks(
                    index, query, visited, notes, budget, expansion
                )
                if not picks:
                    status = self._stalled_status(unread)
                    break

                try:
                    results = await self._read_batch(index, query, picks, notes, budget)
                except (BudgetExhausted, ReasoningParseError):
                    raise
                except Exception as exc:
                    if not steps:
                        raise
                    # A backend that just served N rounds failing on round
                    # N+1 does not retroactively invalidate them. The partial
                    # trace is the product; discarding it here was the bug.
                    logger.warning(
                        "round failed after %d step(s); keeping the partial trace: %s",
                        len(steps),
                        exc,
                    )
                    status = RunStatus.BUDGET_EXHAUSTED
                    break
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

        try:
            answer = await _within_deadline(
                budget, compose(self._model, query=query, notes=notes, budget=budget)
            )
        except Exception as exc:
            # Composing is a nicety; the notes are the substance. A failure
            # here must not discard the steps already gathered.
            logger.warning("composing failed; falling back to the last note: %s", exc)
            answer = notes[-1].finding if notes else NOTHING_FOUND
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
        expansion: tuple[str, ...] | None,
    ) -> tuple[list[_Pick], tuple[str, ...] | None]:
        """The units to read this round, and the expansion to carry forward.

        An escalation, cheapest first:

        1. **trust retrieval** when the ranking is confident;
        2. **let the model write the vocabulary** when it is not — one short
           call, then rank again;
        3. **hedge** when it still is not — read the best lexical guess *and*
           the model's own pick, in the same parallel round.

        Step 3 exists because the previous version asked the wrong question. It
        fell back to the model only when the shortlist was *empty*, so a
        spurious match on a common word — "montant", "fournisseur" — produced a
        non-empty, wrong shortlist and silently suppressed the fallback. What
        matters is not whether retrieval found something but whether what it
        found is worth believing.
        """
        cfg = self._config
        room = min(cfg.fanout, budget.steps_left, budget.calls_left)
        if room <= 0:
            return [], expansion

        probe = self._probe(index, query, notes, visited)
        shortlist = self._rank(index, probe, visited, expansion or ())

        if not self._trusted(shortlist):
            expansion = await self._expanded(index, query, expansion, budget)
            if expansion:
                shortlist = self._rank(index, probe, visited, expansion)

        if self._trusted(shortlist):
            width = min(room, self._fanout_for(shortlist))
            picks = [_Pick(c.unit, c.rationale) for c in shortlist.candidates[:width]]
            return picks, expansion

        hedge_room = self._hedge_room(budget, room)
        picks = await self._hedge(index, query, visited, notes, budget, shortlist, hedge_room)
        return picks, expansion

    @staticmethod
    def _hedge_room(budget: Budget, room: int) -> int:
        """Slots for a hedged round — `fanout` does not get to cancel the hedge.

        `fanout` is about breadth *when retrieval is confident*. A hedge is the
        opposite situation, and clamping it to `fanout` broke the one case it
        exists for: at `fanout = 1` the lexical guess filled the round, the
        model was never asked, and the trace still said "low-confidence
        shortlist". Budget still binds — with a single read left there is no
        second slot to give away.
        """
        return min(max(room, HEDGE_WIDTH), budget.steps_left, budget.calls_left)

    def _rank(
        self,
        index: DocIndex,
        probe: str,
        visited: Sequence[str],
        expansion: Sequence[str],
    ) -> Shortlist:
        if not probe:
            return Shortlist()
        return index.ranker.shortlist(
            probe,
            limit=self._config.shortlist_size,
            exclude=visited,
            expansion=expansion,
        )

    def _trusted(self, shortlist: Shortlist) -> bool:
        return shortlist.confident(
            min_coverage=self._config.min_coverage, min_margin=self._config.min_margin
        )

    async def _expanded(
        self,
        index: DocIndex,
        query: str,
        expansion: tuple[str, ...] | None,
        budget: Budget,
    ) -> tuple[str, ...] | None:
        """Ask the model for retrieval vocabulary — once per run, on demand.

        `None` means "not asked yet"; an empty tuple means "asked, nothing
        usable came back" and is remembered, so a dry answer is never bought
        again on the next low-confidence round. A failed call degrades to the
        hedge instead of aborting the run: the expansion is an enhancement,
        and `expand_query=False` is a supported configuration with exactly
        that behaviour.
        """
        cfg = self._config
        if expansion is not None or not cfg.expand_query or budget.calls_left < 2:
            return expansion
        try:
            terms = await _within_deadline(
                budget,
                expand_query(
                    self._model,
                    query=query,
                    titles=[unit.title for unit in index.units],
                    budget=budget,
                    max_tokens=cfg.max_tokens,
                ),
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("query expansion failed; hedging without it: %s", exc)
            return ()
        if terms:
            logger.info("expanded %r → %s", query[:60], list(terms))
        return terms

    async def _hedge(
        self,
        index: DocIndex,
        query: str,
        visited: Sequence[str],
        notes: Sequence[Note],
        budget: Budget,
        shortlist: Shortlist,
        room: int,
    ) -> list[_Pick]:
        """Read the best lexical guess and the model's pick, side by side.

        The frontier is already parallel, so covering both costs one selection
        call rather than a whole extra round.
        """
        cfg = self._config
        picks: list[_Pick] = []
        base = ""
        top = shortlist.top
        if top is not None:
            base = (
                f"{top.rationale}; low-confidence shortlist "
                f"(coverage {shortlist.coverage:.0%}, margin {shortlist.margin:.2f})"
            )
            picks.append(_Pick(top.unit, base))

        chosen = {pick.unit.ref for pick in picks}
        unread = [u for u in index.units if u.ref not in visited and u.ref not in chosen]
        if not unread or room <= len(picks) or budget.calls_left < len(picks) + 3:
            return picks[:room]

        try:
            selection, fallback = await _within_deadline(
                budget,
                select_unit(
                    self._model,
                    query=query,
                    index=index,
                    candidates=unread,
                    # The round's own picks count as visited for the descent,
                    # or it can re-offer the lexical top and read it twice.
                    visited=[*visited, *chosen],
                    notes=notes,
                    budget=budget,
                    outline_char_budget=cfg.outline_char_budget,
                    max_tokens=cfg.max_tokens,
                ),
            )
        except (BudgetExhausted, ReasoningParseError):
            raise
        except Exception as exc:
            logger.warning("selection failed; hedging with the lexical pick alone: %s", exc)
            return picks[:room]

        unit = index.get(selection.ref)
        if unit is not None and unit.ref not in chosen:
            picks.append(_Pick(unit, selection.reason, fallback=fallback))
            if base:
                # Only now is the claim true. A hedge truncated to the lexical
                # guess must not record a consultation that never happened.
                picks[0] = _Pick(picks[0].unit, f"{base} — the model's own pick is read alongside")
        return picks[:room]

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
        """
        # Guarded because `notes[-0:]` is the whole list, not none of it.
        recent = notes[-self._config.gap_notes :] if self._config.gap_notes > 0 else []
        gap = " ".join(note.finding for note in recent).strip()
        if not visited:
            return query
        if index.ranker.body.missing_terms(query, visited):
            return f"{query} {gap}".strip()
        return gap

    def _fanout_for(self, shortlist: Shortlist) -> int:
        """Breadth when the shortlist is flat, depth when it has a clear winner."""
        if len(shortlist) < 2:
            return 1
        return 1 if shortlist.margin >= self._config.decisive_ratio else self._config.fanout

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
        # The picks were sized when the round started; selection and expansion
        # calls since then may have spent budget. Shrinking here beats raising
        # before a single read has happened.
        batch = list(picks)[: min(len(picks), budget.steps_left, budget.calls_left)]
        if not batch:
            return []
        picks = batch
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
        outcomes = await _within_deadline(budget, asyncio.gather(*tasks, return_exceptions=True))

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
