"""Outline-guided reading loop.

Shape is deliberately close to the upstream chunkless loop — select a unit, read
it, decide — so the emitted trace drops straight into an existing viewer. What
changed is the parts that were making it fragile or expensive:

* **Flat working memory.** Upstream keeps one growing chat session, so every
  section read stays in context and the "chunkless" saving erodes with each hop.
  Here each call carries the query, a fitted outline and short *notes* — the
  prompt stays roughly constant across hops.
* **Cheap path for short documents.** Below a threshold the document is read in
  one call instead of navigated in six.
* **Honest outcomes.** `sufficient` / `absent` / budget give four distinct end
  states instead of one `converged` boolean, so "not in this document" can be
  said out loud rather than guessed at.
* **Revisiting allowed.** A section can become relevant once more is known;
  upstream bans it permanently.
* **Recoverable selection.** An invalid ref gets one corrective round-trip, and
  the fallback is recorded on the step rather than disguised as a real choice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from glosa.domain.budget import Budget
from glosa.domain.errors import BudgetExhausted
from glosa.domain.outline import render_outline
from glosa.domain.values import Excerpt, RunStatus, Span, Step, Trace, UnitKind
from glosa.ports.chat import Message, system, user

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.index import DocIndex, Unit
    from glosa.ports.chat import ChatModel

logger = logging.getLogger(__name__)

_MAX_SPANS_PER_STEP = 200

SYSTEM_PROMPT = (
    "You are a precise research assistant reading one document. "
    "Ground every statement in the text you are shown. "
    "Never invent facts, numbers or citations. "
    "If the document does not contain the answer, say so plainly instead of guessing."
)


class Selection(BaseModel):
    """Which unit to read next."""

    reason: str = Field(description="Why this part of the document should answer the query")
    ref: str = Field(description="Exact ref from the candidate list, e.g. '#/texts/3'")


class Reading(BaseModel):
    """Outcome of reading one unit."""

    sufficient: bool = Field(description="True only if the query can now be fully answered")
    response: str = Field(description="The answer if sufficient, otherwise what is still missing")
    absent: bool = Field(
        description="True if this document plainly does not contain the answer",
        default=False,
    )


@dataclass(frozen=True, slots=True)
class Note:
    """What one read established, kept short on purpose.

    Notes — not raw section text — are what later prompts carry. That is what
    keeps the prompt size roughly constant across hops.
    """

    ref: str
    title: str
    finding: str

    def render(self) -> str:
        return f"[{self.ref} — {self.title}] {self.finding}"


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
            return self._trace(
                query,
                "The document contains no readable content.",
                RunStatus.NOT_IN_DOCUMENT,
                (),
                budget,
                started,
            )

        if index.total_chars <= cfg.direct_char_threshold:
            return await self._run_direct(index, query, budget, started)

        return await self._run_loop(index, units, query, budget, started)

    # -- the two paths --------------------------------------------------------

    async def _run_direct(
        self, index: DocIndex, query: str, budget: Budget, started: float
    ) -> Trace:
        """Short document: one read, no navigation."""
        excerpt = index.full_excerpt(char_budget=self._config.excerpt_char_budget)
        budget.spend_step()
        budget.spend_call()
        reading = await self._read(query, title=index.title, excerpt_text=excerpt.text, notes=())

        status = RunStatus.ANSWERED if reading.sufficient else RunStatus.NOT_IN_DOCUMENT
        if not reading.sufficient and not reading.absent:
            status = RunStatus.INSUFFICIENT_EVIDENCE
        step = Step(
            index=1,
            ref=excerpt.ref,
            reason="Document is short enough to read in full",
            excerpt_chars=len(excerpt),
            sufficient=reading.sufficient,
            response=reading.response,
            kind=UnitKind.DOCUMENT,
            title=index.title,
            pages=excerpt.pages,
            spans=_spans_of(excerpt),
            node_ids=excerpt.node_ids,
        )
        return self._trace(query, reading.response, status, (step,), budget, started)

    async def _run_loop(
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
        answer = ""

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
                selection, fallback = await self._select(
                    query, units, candidates, visited, notes, budget
                )
                unit = index.get(selection.ref)
                if unit is None:  # pragma: no cover - _select guarantees membership
                    break

                excerpt = index.excerpt(unit.ref, char_budget=cfg.excerpt_char_budget)
                budget.spend_call()
                reading = await self._read(
                    query, title=unit.title, excerpt_text=excerpt.text, notes=tuple(notes)
                )

                steps.append(
                    Step(
                        index=len(steps) + 1,
                        ref=unit.ref,
                        reason=selection.reason,
                        excerpt_chars=len(excerpt),
                        sufficient=reading.sufficient,
                        response=reading.response,
                        kind=unit.kind,
                        title=unit.title,
                        pages=excerpt.pages,
                        spans=_spans_of(excerpt),
                        node_ids=excerpt.node_ids,
                        revisited=revisiting or unit.ref in visited,
                        fallback=fallback,
                    )
                )
                visited.append(unit.ref)
                notes.append(Note(unit.ref, unit.title, reading.response[: cfg.note_char_limit]))
                del notes[: max(0, len(notes) - cfg.max_notes)]

                if reading.sufficient:
                    return self._trace(
                        query, reading.response, RunStatus.ANSWERED, tuple(steps), budget, started
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

        answer = await self._compose(query, notes, budget)
        rendered = tuple(note.render() for note in notes)
        return self._trace(query, answer, status, tuple(steps), budget, started, rendered)

    # -- model calls ----------------------------------------------------------

    async def _select(
        self,
        query: str,
        units: Sequence[Unit],
        candidates: Sequence[Unit],
        visited: Sequence[str],
        notes: Sequence[Note],
        budget: Budget,
    ) -> tuple[Selection, bool]:
        """Pick the next unit. Returns `(selection, used_fallback)`."""
        outline = render_outline(
            units, char_budget=self._config.outline_char_budget, visited=visited
        )
        allowed = [u.ref for u in candidates]
        prompt = _selection_prompt(query, outline, allowed, notes)

        messages = [system(SYSTEM_PROMPT), user(prompt)]
        selection = await self._ask_selection(messages, allowed, budget)
        if selection is not None:
            return selection, False

        logger.warning("model returned an unusable ref; falling back to %s", allowed[0])
        return Selection(reason="fallback: no valid ref returned", ref=allowed[0]), True

    async def _ask_selection(
        self, messages: list[Message], allowed: Sequence[str], budget: Budget
    ) -> Selection | None:
        allowed_set = frozenset(allowed)
        budget.spend_call()
        attempt = await self._model.structured(
            messages, schema=Selection, max_tokens=self._config.max_tokens
        )
        if attempt.ref in allowed_set:
            return attempt

        # One corrective round-trip naming the mistake — cheaper and far more
        # reliable than rejection-sampling the same prompt three times.
        correction = [
            *messages,
            Message("assistant", attempt.model_dump_json()),
            user(
                f"'{attempt.ref}' is not in the candidate list. "
                f"Answer again choosing exactly one of: {list(allowed)}"
            ),
        ]
        budget.spend_call()
        retry = await self._model.structured(
            correction, schema=Selection, max_tokens=self._config.max_tokens
        )
        return retry if retry.ref in allowed_set else None

    async def _read(
        self, query: str, *, title: str, excerpt_text: str, notes: Sequence[Note]
    ) -> Reading:
        prompt = _reading_prompt(query, title, excerpt_text, notes)
        return await self._model.structured(
            [system(SYSTEM_PROMPT), user(prompt)],
            schema=Reading,
            max_tokens=self._config.max_tokens,
        )

    async def _compose(self, query: str, notes: Sequence[Note], budget: Budget) -> str:
        """Assemble the best partial answer from the notes gathered so far."""
        if not notes:
            return "No relevant content was found in this document."
        if len(notes) == 1 or not budget.can_afford_extra_call():
            return notes[-1].finding

        joined = "\n\n".join(note.render() for note in notes)
        prompt = (
            f"Question: {query}\n\n"
            f"Notes gathered while reading the document:\n{joined}\n\n"
            "Write the most complete answer these notes support. State explicitly "
            "what remains unknown. Do not add anything the notes do not contain."
        )
        budget.spend_call()
        return (await self._model.complete([system(SYSTEM_PROMPT), user(prompt)])).strip()

    # -- assembly -------------------------------------------------------------

    def _trace(
        self,
        query: str,
        answer: str,
        status: RunStatus,
        steps: tuple[Step, ...],
        budget: Budget,
        started: float,
        notes: Sequence[str] = (),
    ) -> Trace:
        return Trace(
            query=query,
            answer=answer,
            status=status,
            model_id=self._model.model_id,
            steps=steps,
            llm_calls=budget.calls,
            elapsed_s=round(time.monotonic() - started, 3),
            notes=tuple(notes),
        )


def _render_notes(notes: Sequence[Note]) -> str:
    return "\n".join(note.render() for note in notes) if notes else "(nothing read yet)"


def _spans_of(excerpt: Excerpt) -> tuple[Span, ...]:
    return tuple(
        Span(self_ref=p.self_ref, node_id=p.node_id, page_no=p.page_no, bbox=p.bbox)
        for p in excerpt.parts[:_MAX_SPANS_PER_STEP]
    )


def _selection_prompt(
    query: str, outline: str, allowed: Sequence[str], notes: Sequence[Note]
) -> str:
    known = _render_notes(notes)
    return (
        f"Question: {query}\n\n"
        f"Document map (ref · kind · size · heading):\n{outline}\n\n"
        f"What you already know:\n{known}\n\n"
        f"Choose the ONE part most likely to advance the answer.\n"
        f"Its ref must be exactly one of: {list(allowed)}"
    )


def _reading_prompt(query: str, title: str, excerpt: str, notes: Sequence[Note]) -> str:
    known = _render_notes(notes)
    body = excerpt or "(this part of the document is empty)"
    return (
        f"Question: {query}\n\n"
        f"What you already know:\n{known}\n\n"
        f"Now read this part of the document — '{title}':\n\n{body}\n\n"
        "Decide:\n"
        "- sufficient: true only if you can now answer the question completely, "
        "using only what you have read.\n"
        "- response: the complete answer if sufficient; otherwise state precisely "
        "what is still missing.\n"
        "- absent: true only if this document clearly is not about the subject of "
        "the question at all."
    )
