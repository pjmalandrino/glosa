"""The reading loop, driven by a scripted model."""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.base import Size

from glosa.document.index import DocIndex
from glosa.strategy.navigate import NavigateConfig, NavigateStrategy, Reading, Selection
from glosa.types import RunStatus, UnitKind
from tests.conftest import FakeChatModel

LOOP = NavigateConfig(direct_char_threshold=0)


def _refs(doc: DoclingDocument) -> list[str]:
    return [u.ref for u in DocIndex(doc).units]


def _long_doc() -> DoclingDocument:
    """Two sections whose bodies are long and mutually distinctive."""
    doc = DoclingDocument(name="contract")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    scope = doc.add_heading(text="Scope", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="SCOPETEXT " * 400, parent=scope)
    penalties = doc.add_heading(text="Penalties", level=1)
    doc.add_text(
        label=DocItemLabel.TEXT, text="Late delivery incurs 2% per week.", parent=penalties
    )
    return doc


# -- the cheap path -----------------------------------------------------------


async def test_a_short_document_is_read_in_one_call(flat_doc: DoclingDocument) -> None:
    """No navigation loop for a document that fits — one call, not six."""
    model = FakeChatModel([Reading(sufficient=True, response="12.4M EUR.")])
    trace = await NavigateStrategy(model).run(DocIndex(flat_doc), "What was revenue?")

    assert trace.status is RunStatus.ANSWERED
    assert trace.converged is True
    assert trace.llm_calls == 1
    assert len(trace.steps) == 1
    assert trace.steps[0].kind is UnitKind.DOCUMENT
    assert trace.steps[0].ref == "#/body"


# -- the loop -----------------------------------------------------------------


async def test_the_loop_converges_and_records_every_hop(flat_doc: DoclingDocument) -> None:
    refs = _refs(flat_doc)
    model = FakeChatModel(
        [
            Selection(reason="revenue lives here", ref=refs[1]),
            Reading(sufficient=False, response="Found the heading, not the figure."),
            Selection(reason="try the risks section", ref=refs[2]),
            Reading(sufficient=True, response="Revenue was 12.4M EUR."),
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "revenue?")

    assert trace.status is RunStatus.ANSWERED
    assert trace.answer == "Revenue was 12.4M EUR."
    assert [s.ref for s in trace.steps] == [refs[1], refs[2]]
    assert [s.sufficient for s in trace.steps] == [False, True]
    assert trace.steps[0].reason == "revenue lives here"
    assert trace.llm_calls == 4


async def test_context_stays_flat_across_hops() -> None:
    """Upstream keeps one growing session, so every section read stays in
    context and the chunkless saving erodes. Here later prompts carry a short
    note, never the previous section's body."""
    doc = _long_doc()
    refs = _refs(doc)
    model = FakeChatModel(
        [
            Selection(reason="start at scope", ref=refs[0]),
            Reading(sufficient=False, response="Scope defines the works only."),
            Selection(reason="penalties next", ref=refs[1]),
            Reading(sufficient=True, response="2% per week."),
        ]
    )
    await NavigateStrategy(model, LOOP).run(DocIndex(doc), "penalty rate?")

    first_read = model.structured_calls[1][-1].content
    assert "SCOPETEXT" in first_read, "step 1 must actually show the section"

    for later in model.structured_calls[2:]:
        body = later[-1].content
        assert "SCOPETEXT" not in body, "section text must not be carried forward"
        assert "Scope defines the works only." in body, "the note must be"


async def test_an_invalid_ref_gets_one_corrective_round_trip(
    flat_doc: DoclingDocument,
) -> None:
    refs = _refs(flat_doc)
    model = FakeChatModel(
        [
            Selection(reason="guessing", ref="#/texts/999"),
            Selection(reason="corrected", ref=refs[1]),
            Reading(sufficient=True, response="ok"),
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "q")

    assert trace.steps[0].ref == refs[1]
    assert trace.steps[0].fallback is False
    correction = model.structured_calls[1][-1].content
    assert "#/texts/999" in correction and "is not in the candidate list" in correction


async def test_a_persistently_invalid_ref_falls_back_and_says_so(
    flat_doc: DoclingDocument,
) -> None:
    """Upstream silently records `reason='fallback'` as if it were a choice."""
    model = FakeChatModel(
        [
            Selection(reason="nope", ref="#/texts/999"),
            Selection(reason="still nope", ref="#/texts/998"),
            Reading(sufficient=True, response="ok"),
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "q")

    assert trace.steps[0].fallback is True
    assert trace.steps[0].ref in _refs(flat_doc)


async def test_two_absent_votes_end_the_run_honestly(flat_doc: DoclingDocument) -> None:
    """`not_in_document` is an outcome, not a failure to converge."""
    refs = _refs(flat_doc)
    model = FakeChatModel(
        [
            Selection(reason="a", ref=refs[1]),
            Reading(sufficient=False, response="Not about cats.", absent=True),
            Selection(reason="b", ref=refs[2]),
            Reading(sufficient=False, response="Still not about cats.", absent=True),
            "This document does not discuss cats.",
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "cats?")

    assert trace.status is RunStatus.NOT_IN_DOCUMENT
    assert trace.converged is False
    assert len(trace.steps) == 2


async def test_one_absent_vote_is_not_enough(flat_doc: DoclingDocument) -> None:
    refs = _refs(flat_doc)
    model = FakeChatModel(
        [
            Selection(reason="a", ref=refs[1]),
            Reading(sufficient=False, response="Nothing here.", absent=True),
            Selection(reason="b", ref=refs[2]),
            Reading(sufficient=True, response="Actually, here it is."),
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "q")
    assert trace.status is RunStatus.ANSWERED


async def test_running_out_of_steps_with_material_left_is_a_budget_outcome() -> None:
    doc = _long_doc()
    refs = _refs(doc)
    model = FakeChatModel(
        [
            Selection(reason="a", ref=refs[0]),
            Reading(sufficient=False, response="partial finding"),
        ]
    )
    config = NavigateConfig(direct_char_threshold=0, max_steps=1)
    trace = await NavigateStrategy(model, config).run(DocIndex(doc), "q")

    assert trace.status is RunStatus.BUDGET_EXHAUSTED
    assert trace.answer == "partial finding"


async def test_reading_everything_without_an_answer_is_an_evidence_outcome(
    flat_doc: DoclingDocument,
) -> None:
    refs = _refs(flat_doc)
    script: list[object] = []
    for ref in refs:
        script.append(Selection(reason="looking", ref=ref))
        script.append(Reading(sufficient=False, response=f"nothing in {ref}"))
    script.append("Nothing conclusive was found.")

    config = NavigateConfig(direct_char_threshold=0, max_steps=len(refs), allow_revisit=False)
    trace = await NavigateStrategy(model := FakeChatModel(script), config).run(
        DocIndex(flat_doc), "q"
    )

    assert trace.status is RunStatus.INSUFFICIENT_EVIDENCE
    assert trace.answer == "Nothing conclusive was found."
    assert len(trace.steps) == len(refs)
    assert model.complete_calls, "a partial answer should be composed from the notes"


async def test_a_single_note_is_returned_without_an_extra_call() -> None:
    doc = _long_doc()
    model = FakeChatModel(
        [
            Selection(reason="a", ref=_refs(doc)[0]),
            Reading(sufficient=False, response="only this"),
        ]
    )
    config = NavigateConfig(direct_char_threshold=0, max_steps=1)
    trace = await NavigateStrategy(model, config).run(DocIndex(doc), "q")

    assert trace.answer == "only this"
    assert model.complete_calls == []


async def test_the_llm_call_ceiling_stops_the_run() -> None:
    doc = _long_doc()
    refs = _refs(doc)
    model = FakeChatModel(
        [
            Selection(reason="a", ref=refs[0]),
            Reading(sufficient=False, response="partial"),
            Selection(reason="b", ref=refs[1]),
        ]
    )
    config = NavigateConfig(direct_char_threshold=0, max_steps=5, max_llm_calls=3)
    trace = await NavigateStrategy(model, config).run(DocIndex(doc), "q")

    assert trace.status is RunStatus.BUDGET_EXHAUSTED
    assert trace.llm_calls <= 3


async def test_provenance_travels_with_every_step(flat_doc: DoclingDocument) -> None:
    refs = _refs(flat_doc)
    model = FakeChatModel(
        [
            Selection(reason="a", ref=refs[1]),
            Reading(sufficient=True, response="done"),
        ]
    )
    trace = await NavigateStrategy(model, LOOP).run(DocIndex(flat_doc), "q")

    step = trace.steps[0]
    assert step.pages == (1,)
    assert step.spans and step.spans[0].self_ref.startswith("#/texts/")
    assert any(s.bbox is not None for s in step.spans)


async def test_an_empty_document_is_reported_not_answered() -> None:
    doc = DoclingDocument(name="empty")
    trace = await NavigateStrategy(FakeChatModel([])).run(DocIndex(doc), "q")
    assert trace.status is RunStatus.NOT_IN_DOCUMENT
    assert trace.steps == ()
