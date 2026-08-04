"""Vocabulary shared by the reading strategies.

Both strategies do the same three things — pick something to read, read it,
decide whether that is enough — and differ only in how they pick. Keeping the
prompts, schemas and step construction here means the two cannot drift into
producing subtly different traces for the same document.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from glosa.domain.outline import render_outline
from glosa.domain.values import RunStatus, Span, Step, Trace, UnitKind
from glosa.ports.chat import Message, system, user

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.budget import Budget
    from glosa.domain.index import DocIndex, Unit
    from glosa.domain.values import Excerpt
    from glosa.ports.chat import ChatModel

logger = logging.getLogger(__name__)

MAX_SPANS_PER_STEP = 200

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


def render_notes(notes: Sequence[Note]) -> str:
    return "\n".join(note.render() for note in notes) if notes else "(nothing read yet)"


def spans_of(excerpt: Excerpt) -> tuple[Span, ...]:
    return tuple(
        Span(self_ref=p.self_ref, node_id=p.node_id, page_no=p.page_no, bbox=p.bbox)
        for p in excerpt.parts[:MAX_SPANS_PER_STEP]
    )


def make_step(
    *,
    index: int,
    unit: Unit,
    excerpt: Excerpt,
    reason: str,
    reading: Reading,
    revisited: bool = False,
    fallback: bool = False,
) -> Step:
    """Build the trace entry for one read."""
    return Step(
        index=index,
        ref=unit.ref,
        reason=reason,
        excerpt_chars=len(excerpt),
        sufficient=reading.sufficient,
        response=reading.response,
        kind=unit.kind,
        title=unit.title,
        pages=excerpt.pages,
        spans=spans_of(excerpt),
        node_ids=excerpt.node_ids,
        revisited=revisited,
        fallback=fallback,
    )


def selection_prompt(
    query: str, outline: str, allowed: Sequence[str], notes: Sequence[Note]
) -> str:
    return (
        f"Question: {query}\n\n"
        f"Document map (ref · kind · size · heading):\n{outline}\n\n"
        f"What you already know:\n{render_notes(notes)}\n\n"
        f"Choose the ONE part most likely to advance the answer.\n"
        f"Its ref must be exactly one of: {list(allowed)}"
    )


def reading_prompt(query: str, title: str, excerpt: str, notes: Sequence[Note]) -> str:
    body = excerpt or "(this part of the document is empty)"
    return (
        f"Question: {query}\n\n"
        f"What you already know:\n{render_notes(notes)}\n\n"
        f"Now read this part of the document — '{title}':\n\n{body}\n\n"
        "Decide:\n"
        "- sufficient: true only if you can now answer the question completely, "
        "using only what you have read.\n"
        "- response: the complete answer if sufficient; otherwise state precisely "
        "what is still missing.\n"
        "- absent: true only if this document clearly is not about the subject of "
        "the question at all."
    )


def compose_prompt(query: str, notes: Sequence[Note]) -> str:
    joined = "\n\n".join(note.render() for note in notes)
    return (
        f"Question: {query}\n\n"
        f"Notes gathered while reading the document:\n{joined}\n\n"
        "Write the most complete answer these notes support. State explicitly "
        "what remains unknown. Do not add anything the notes do not contain."
    )


# --- the three model calls a strategy can make --------------------------------


async def read_unit(
    model: ChatModel,
    *,
    query: str,
    title: str,
    excerpt_text: str,
    notes: Sequence[Note],
    max_tokens: int | None,
) -> Reading:
    """Read one unit and decide whether it settles the question."""
    return await model.structured(
        [system(SYSTEM_PROMPT), user(reading_prompt(query, title, excerpt_text, notes))],
        schema=Reading,
        max_tokens=max_tokens,
    )


async def select_unit(
    model: ChatModel,
    *,
    query: str,
    index: DocIndex,
    candidates: Sequence[Unit],
    visited: Sequence[str],
    notes: Sequence[Note],
    budget: Budget,
    outline_char_budget: int,
    max_tokens: int | None,
) -> tuple[Selection, bool]:
    """Ask the model to pick the next unit. Returns `(selection, used_fallback)`.

    Two things the naive version gets wrong:

    * it offers refs the map does not show. When the outline was fitted to a
      budget, the candidate list must shrink to what was actually rendered —
      otherwise the model is invited to name a section it never saw.
    * it navigates a flat list. When the fitted map hid a section's
      subsections, picking that section opens a second, cheap round over just
      those children: coarse-to-fine, bounded to one extra call.
    """
    outline = render_outline(index.units, char_budget=outline_char_budget, visited=visited)
    visible = [u for u in candidates if u.ref in outline.refs] or list(candidates)

    selection, fallback = await _choose(
        model,
        query=query,
        outline=outline.text,
        candidates=visible,
        notes=notes,
        budget=budget,
        max_tokens=max_tokens,
    )
    if fallback:
        return selection, True

    hidden = [
        child
        for child in index.children_of(selection.ref)
        if child.ref not in outline.refs and child.ref not in visited
    ]
    if not hidden or budget.calls_left < 1:
        return selection, False

    parent = index.get(selection.ref)
    deeper_candidates = [*([parent] if parent is not None else []), *hidden]
    deeper_outline = render_outline(
        deeper_candidates, char_budget=outline_char_budget, visited=visited
    )
    deeper, deeper_fallback = await _choose(
        model,
        query=query,
        outline=deeper_outline.text,
        candidates=deeper_candidates,
        notes=notes,
        budget=budget,
        max_tokens=max_tokens,
    )
    # A failed descent is not a failed selection: keep the parent.
    return (selection if deeper_fallback else deeper), False


async def _choose(
    model: ChatModel,
    *,
    query: str,
    outline: str,
    candidates: Sequence[Unit],
    notes: Sequence[Note],
    budget: Budget,
    max_tokens: int | None,
) -> tuple[Selection, bool]:
    """One pick over one map.

    An unusable ref gets one corrective round-trip naming the mistake — cheaper
    and far more reliable than rejection-sampling the same prompt three times.
    If that also fails, the first candidate is taken and the caller records that
    glosa chose, rather than dressing it up as the model's decision.
    """
    allowed = [u.ref for u in candidates]
    allowed_set = frozenset(allowed)
    messages = [system(SYSTEM_PROMPT), user(selection_prompt(query, outline, allowed, notes))]

    budget.spend_call()
    attempt = await model.structured(messages, schema=Selection, max_tokens=max_tokens)
    if attempt.ref in allowed_set:
        return attempt, False

    correction = [
        *messages,
        Message("assistant", attempt.model_dump_json()),
        user(
            f"'{attempt.ref}' is not in the candidate list. "
            f"Answer again choosing exactly one of: {list(allowed)}"
        ),
    ]
    budget.spend_call()
    retry = await model.structured(correction, schema=Selection, max_tokens=max_tokens)
    if retry.ref in allowed_set:
        return retry, False

    logger.warning("model returned an unusable ref; falling back to %s", allowed[0])
    return Selection(reason="fallback: no valid ref returned", ref=allowed[0]), True


async def compose(model: ChatModel, *, query: str, notes: Sequence[Note], budget: Budget) -> str:
    """Best partial answer the notes support."""
    if not notes:
        return "No relevant content was found in this document."
    if len(notes) == 1 or not budget.can_afford_extra_call():
        return notes[-1].finding

    budget.spend_call()
    reply = await model.complete([system(SYSTEM_PROMPT), user(compose_prompt(query, notes))])
    return reply.strip()


async def read_whole_document(
    model: ChatModel,
    *,
    index: DocIndex,
    query: str,
    budget: Budget,
    excerpt_char_budget: int,
    max_tokens: int | None,
) -> tuple[Step, RunStatus]:
    """The cheap path: a document that fits is read once, not navigated.

    Anchored on the first projected element so the step still points at a node
    the host's graph has, and carrying the node ids of everything read.
    """
    excerpt = index.full_excerpt(char_budget=excerpt_char_budget)
    budget.spend_step()
    budget.spend_call()
    reading = await read_unit(
        model,
        query=query,
        title=index.title,
        excerpt_text=excerpt.text,
        notes=(),
        max_tokens=max_tokens,
    )

    if reading.sufficient:
        status = RunStatus.ANSWERED
    elif reading.absent:
        status = RunStatus.NOT_IN_DOCUMENT
    else:
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
        spans=spans_of(excerpt),
        node_ids=excerpt.node_ids,
    )
    return step, status


def build_trace(
    *,
    query: str,
    answer: str,
    status: RunStatus,
    steps: Sequence[Step],
    budget: Budget,
    started: float,
    model_id: str,
    notes: Sequence[Note] = (),
) -> Trace:
    return Trace(
        query=query,
        answer=answer,
        status=status,
        model_id=model_id,
        steps=tuple(steps),
        llm_calls=budget.calls,
        elapsed_s=round(time.monotonic() - started, 3),
        notes=tuple(note.render() for note in notes),
    )


class QueryTerms(BaseModel):
    """Retrieval vocabulary for one question, written by the model."""

    terms: list[str] = Field(
        description=(
            "4 to 8 words or short phrases that would plausibly appear IN THE DOCUMENT "
            "in the passage answering the question. Use the document's own language and "
            "register, not a rephrasing of the question."
        )
    )


MAX_EXPANSION_TERMS = 10


async def expand_query(
    model: ChatModel,
    *,
    query: str,
    titles: Sequence[str],
    budget: Budget,
    max_tokens: int | None,
) -> tuple[str, ...]:
    """Turn a question into vocabulary the document might actually use.

    A lexical prior fails on paraphrase: the user asks for "le montant maximum
    que le fournisseur devra rembourser" and the document says "plafond
    d'indemnisation". Nothing about term matching closes that gap; a model
    closes it in one short call, and writing vocabulary is a generation task
    small models are good at — unlike choosing from a long outline.

    The section headings are shown as a register hint. They are often
    uninformative ("Article 7"), in which case they are simply ignored.
    """
    hint = _clip_titles(titles)
    prompt = (
        f"Document sections:\n{hint}\n\n"
        f"Question: {query}\n\n"
        "List the words and short phrases most likely to appear in the passage "
        "of this document that answers the question. Write them as the document "
        "would write them, in its language. Do not rephrase the question."
    )
    budget.spend_call()
    reply = await model.structured(
        [system(SYSTEM_PROMPT), user(prompt)], schema=QueryTerms, max_tokens=max_tokens
    )
    seen: dict[str, None] = {}
    for term in reply.terms:
        cleaned = " ".join(str(term).split())
        if cleaned:
            seen[cleaned] = None
    return tuple(seen)[:MAX_EXPANSION_TERMS]


def _clip_titles(titles: Sequence[str], limit: int = 600) -> str:
    lines: list[str] = []
    used = 0
    for title in titles:
        line = f"- {title}"
        if used + len(line) > limit:
            lines.append(f"- … ({len(titles) - len(lines)} more)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(no headings)"
