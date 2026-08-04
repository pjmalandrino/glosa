"""The loop glosa replaced, kept only to measure against.

`docling-agent`'s `_rag_loop` and glosa's own first strategy have the same
shape: ask the model to pick a section from the map, ask it to read that
section, repeat. Two LLM calls per section, always.

It is not shipped. A package with two reading loops has two ways to answer the
same question, and they drift. But deleting it outright would also delete the
baseline behind "one call instead of four", and that number is the point of the
retrieval prior — so it lives here, in the tests, which is what a baseline is.

It reuses `reading.py`'s calls, so the comparison is between *loops*, not
between two different sets of prompts.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from glosa.domain.budget import Budget
from glosa.domain.reading import (
    Note,
    build_trace,
    make_step,
    read_unit,
    select_unit,
)
from glosa.domain.values import RunStatus

if TYPE_CHECKING:
    from glosa.domain.index import DocIndex
    from glosa.domain.values import Step, Trace
    from glosa.ports.chat import ChatModel


async def run_baseline(
    model: ChatModel, index: DocIndex, query: str, *, max_steps: int = 6
) -> Trace:
    """Select → read → decide, one section per hop, no retrieval prior."""
    started = time.monotonic()
    budget = Budget(max_steps=max_steps, max_llm_calls=20, deadline_s=None)
    steps: list[Step] = []
    notes: list[Note] = []
    visited: list[str] = []

    while budget.steps_left:
        candidates = [u for u in index.units if u.ref not in visited]
        if not candidates:
            break
        budget.spend_step()
        selection, fallback = await select_unit(
            model,
            query=query,
            index=index,
            candidates=candidates,
            visited=visited,
            notes=notes,
            budget=budget,
            outline_char_budget=6_000,
            max_tokens=None,
        )
        unit = index.get(selection.ref)
        assert unit is not None, "select_unit guarantees a candidate ref"

        excerpt = index.excerpt(unit.ref, char_budget=8_000, focus=query)
        budget.spend_call()
        reading = await read_unit(
            model,
            query=query,
            title=unit.title,
            excerpt_text=excerpt.text,
            notes=tuple(notes),
            max_tokens=None,
        )
        steps.append(
            make_step(
                index=len(steps) + 1,
                unit=unit,
                excerpt=excerpt,
                reason=selection.reason,
                reading=reading,
                fallback=fallback,
            )
        )
        visited.append(unit.ref)
        notes.append(Note(unit.ref, unit.title, reading.response))

        if reading.sufficient:
            return build_trace(
                query=query,
                answer=reading.response,
                status=RunStatus.ANSWERED,
                steps=steps,
                budget=budget,
                started=started,
                model_id=model.model_id,
            )

    return build_trace(
        query=query,
        answer=notes[-1].finding if notes else "",
        status=RunStatus.BUDGET_EXHAUSTED,
        steps=steps,
        budget=budget,
        started=started,
        model_id=model.model_id,
    )
