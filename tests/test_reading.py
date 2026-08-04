"""The three model calls every read is built from.

These used to be exercised through a second reading strategy. Testing them
directly is both smaller and stricter: a strategy can mask a broken
`select_unit` by never reaching the branch that uses it.
"""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument

from glosa.domain.budget import Budget
from glosa.domain.outline import render_outline
from glosa.domain.reading import (
    Note,
    Reading,
    Selection,
    compose,
    make_step,
    read_whole_document,
    select_unit,
)
from glosa.domain.values import RunStatus, UnitKind
from tests.conftest import FakeChatModel, index_of, pages, prov

BUDGET_ARGS = {"outline_char_budget": 6_000, "max_tokens": None}


def _budget(**overrides: object) -> Budget:
    base: dict[str, object] = {"max_steps": 6, "max_llm_calls": 20, "deadline_s": None}
    base.update(overrides)
    return Budget(**base)  # type: ignore[arg-type]


async def _select(model: FakeChatModel, index, query: str, **overrides):  # type: ignore[no-untyped-def]
    kwargs = {**BUDGET_ARGS, **overrides}
    return await select_unit(
        model,
        query=query,
        index=index,
        candidates=list(index.units),
        visited=[],
        notes=[],
        budget=_budget(),
        **kwargs,  # type: ignore[arg-type]
    )


# -- select_unit: choosing over a fitted map -----------------------------------


async def test_an_invalid_ref_gets_one_corrective_round_trip(flat_json: str) -> None:
    """Naming the mistake is cheaper and far more reliable than resampling."""
    index = index_of(flat_json)
    good = index.units[1].ref
    model = FakeChatModel(
        [
            Selection(reason="guessing", ref="#/texts/999"),
            Selection(reason="corrected", ref=good),
        ]
    )

    selection, fallback = await _select(model, index, "q")

    assert selection.ref == good
    assert fallback is False
    correction = model.structured_calls[1][-1].content
    assert "#/texts/999" in correction and "is not in the candidate list" in correction


async def test_a_persistently_invalid_ref_falls_back_and_says_so(flat_json: str) -> None:
    """Upstream silently records `reason='fallback'` as if it were a choice."""
    index = index_of(flat_json)
    model = FakeChatModel(
        [
            Selection(reason="nope", ref="#/texts/999"),
            Selection(reason="still nope", ref="#/texts/998"),
        ]
    )

    selection, fallback = await _select(model, index, "q")

    assert fallback is True
    assert selection.ref in index.refs


def _flat_sections_json(n: int = 12) -> str:
    """Same-level sections: nothing nests, so no descent round muddies the test."""
    doc = DoclingDocument(name="marche")
    pages(doc, 1)
    for i in range(n):
        heading = doc.add_heading(text=f"Article {i + 1} on reporting", level=1, prov=prov(1, 740))
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Body of article {i}. " * 20,
            parent=heading,
            prov=prov(1, 700),
        )
    return doc.model_dump_json()


async def test_the_model_is_only_offered_refs_the_map_showed() -> None:
    """Offering a ref the map elided invites the model to name what it cannot see."""
    index = index_of(_flat_sections_json())
    shown = render_outline(index.units, char_budget=300).refs
    assert shown < index.refs, "the budget has to actually hide something"
    pick = next(u.ref for u in index.units if u.ref in shown)
    model = FakeChatModel([Selection(reason="a", ref=pick)])

    await _select(model, index, "q", outline_char_budget=300)

    offered = model.structured_calls[0][-1].content.rsplit("exactly one of: ", 1)[1]
    assert all(f"'{ref}'" not in offered for ref in index.refs - shown), "hidden refs offered"
    assert f"'{pick}'" in offered


# -- select_unit: coarse to fine -----------------------------------------------


def _nested_json() -> str:
    doc = DoclingDocument(name="code")
    pages(doc, 1)
    parent = doc.add_heading(text="Chapter 4 on regulatory reporting", level=1, prov=prov(1, 740))
    doc.add_text(
        label=DocItemLabel.TEXT, text="Chapter preamble. " * 40, parent=parent, prov=prov(1, 700)
    )
    child = doc.add_heading(
        text="Section 4.3 on indemnities", level=2, parent=parent, prov=prov(1, 600)
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="The cap is 500,000 EUR. " * 40,
        parent=child,
        prov=prov(1, 580),
    )
    return doc.model_dump_json()


async def test_a_hidden_subsection_gets_a_descent_round() -> None:
    """Pick the chapter from a fitted map, then its subsection — one extra call."""
    index = index_of(_nested_json())
    parent_ref, child_ref = index.units[0].ref, index.units[1].ref
    assert index.children_of(parent_ref) == (index.units[1],)

    model = FakeChatModel(
        [
            Selection(reason="the chapter", ref=parent_ref),
            Selection(reason="its indemnity subsection", ref=child_ref),
        ]
    )
    # A budget so small that only level-0 headings survive the map.
    selection, fallback = await _select(model, index, "indemnity cap", outline_char_budget=90)

    assert selection.ref == child_ref
    assert selection.reason == "its indemnity subsection"
    assert fallback is False


async def test_a_failed_descent_keeps_the_parent() -> None:
    index = index_of(_nested_json())
    parent_ref = index.units[0].ref
    model = FakeChatModel(
        [
            Selection(reason="the chapter", ref=parent_ref),
            Selection(reason="nonsense", ref="#/texts/999"),
            Selection(reason="still nonsense", ref="#/texts/998"),
        ]
    )

    selection, fallback = await _select(model, index, "q", outline_char_budget=90)

    assert selection.ref == parent_ref
    assert fallback is False, "a failed descent is not a failed selection"


# -- read_whole_document: the cheap path ---------------------------------------


async def test_a_short_document_is_read_in_one_call(flat_json: str) -> None:
    """No loop for a document that fits — one call, anchored on a real node."""
    index = index_of(flat_json)
    budget = _budget()
    model = FakeChatModel([Reading(sufficient=True, response="12.4M EUR.")])

    step, status = await read_whole_document(
        model,
        index=index,
        query="revenue?",
        budget=budget,
        excerpt_char_budget=8_000,
        max_tokens=None,
    )

    assert status is RunStatus.ANSWERED
    assert budget.calls == 1
    assert step.kind is UnitKind.DOCUMENT
    assert step.ref in {e.self_ref for e in index.projection.elements}
    assert set(step.node_ids) == set(index.projection.node_ids)


async def test_a_short_document_that_answers_nothing_says_so(flat_json: str) -> None:
    model = FakeChatModel([Reading(sufficient=False, response="not about cats", absent=True)])

    _, status = await read_whole_document(
        model,
        index=index_of(flat_json),
        query="cats?",
        budget=_budget(),
        excerpt_char_budget=8_000,
        max_tokens=None,
    )

    assert status is RunStatus.NOT_IN_DOCUMENT


# -- compose: the partial answer -----------------------------------------------


async def test_a_single_note_is_returned_without_an_extra_call() -> None:
    model = FakeChatModel([])
    notes = [Note("#/texts/0", "Scope", "only this")]

    answer = await compose(model, query="q", notes=notes, budget=_budget())

    assert answer == "only this"
    assert model.complete_calls == [], "one note needs no model to summarize it"


async def test_several_notes_are_composed_into_one_answer() -> None:
    model = FakeChatModel(["Nothing conclusive was found."])
    notes = [Note("#/texts/0", "Scope", "a"), Note("#/texts/2", "Penalties", "b")]

    answer = await compose(model, query="q", notes=notes, budget=_budget())

    assert answer == "Nothing conclusive was found."
    assert "a" in model.complete_calls[0][-1].content


async def test_no_notes_is_reported_as_nothing_found() -> None:
    answer = await compose(FakeChatModel([]), query="q", notes=[], budget=_budget())
    assert "No relevant content" in answer


async def test_composing_is_skipped_when_the_call_budget_is_spent() -> None:
    """A partial answer is worth having; a crash on the way to it is not."""
    model = FakeChatModel([])
    notes = [Note("#/texts/0", "Scope", "a"), Note("#/texts/2", "Penalties", "b")]

    answer = await compose(model, query="q", notes=notes, budget=_budget(max_llm_calls=0))

    assert answer == "b"
    assert model.complete_calls == []


# -- make_step: what the viewer gets -------------------------------------------


def test_provenance_travels_with_every_step(flat_json: str) -> None:
    index = index_of(flat_json)
    unit = index.units[1]
    excerpt = index.excerpt(unit.ref)

    step = make_step(
        index=1,
        unit=unit,
        excerpt=excerpt,
        reason="because",
        reading=Reading(sufficient=True, response="done"),
    )

    assert step.pages == (1,)
    assert step.spans and step.spans[0].self_ref.startswith("#/texts/")
    assert any(s.bbox is not None for s in step.spans)
    assert step.node_ids == excerpt.node_ids
