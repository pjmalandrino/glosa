"""Retrieval-first reading with a parallel frontier."""

from __future__ import annotations

import asyncio

from docling_core.types.doc import DocItemLabel, DoclingDocument

from glosa.domain.hybrid import HybridConfig, HybridStrategy
from glosa.domain.navigate import NavigateConfig, NavigateStrategy
from glosa.domain.reading import Reading, Selection
from glosa.domain.values import RunStatus
from tests.conftest import FakeChatModel, index_of, pages, prov

WIDE = 1_000.0
"""A decisiveness threshold nothing reaches, so fanout is purely the config."""

LOOP = HybridConfig(direct_char_threshold=0)


def _contract_json(*, filler: int = 40) -> str:
    """Three sections with distinct vocabulary, long enough to skip the cheap path."""
    doc = DoclingDocument(name="service-agreement")
    pages(doc, 1)
    sections = [
        ("Scope of works", "The supplier shall deliver the works described in Annex A. " * filler),
        (
            "Liquidated damages",
            "Late delivery incurs a penalty of 2% per week, capped at 10%. " * filler,
        ),
        ("Invoicing", "Invoices are payable within 30 days of receipt. " * filler),
    ]
    for title, body in sections:
        heading = doc.add_heading(text=title, level=1, prov=prov(1, 740))
        doc.add_text(label=DocItemLabel.TEXT, text=body, parent=heading, prov=prov(1, 700))
    return doc.model_dump_json()


def _refs(document_json: str) -> list[str]:
    return [u.ref for u in index_of(document_json).units]


# -- retrieval does the choosing ----------------------------------------------


async def test_retrieval_picks_the_right_section_without_spending_a_call() -> None:
    """One call per unit read, versus two when the model also has to choose."""
    document_json = _contract_json()
    model = FakeChatModel([Reading(sufficient=True, response="2% per week, capped at 10%.")])
    config = HybridConfig(direct_char_threshold=0, fanout=1)

    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "late delivery penalty"
    )

    assert trace.status is RunStatus.ANSWERED
    assert trace.llm_calls == 1
    assert trace.steps[0].title == "Liquidated damages"
    assert "lexical match" in trace.steps[0].reason


async def test_the_same_question_costs_navigate_four_calls() -> None:
    """The comparison the retrieval prior is there to win."""
    document_json = _contract_json()
    refs = _refs(document_json)
    model = FakeChatModel(
        [
            Selection(reason="try scope", ref=refs[0]),
            Reading(sufficient=False, response="Scope only describes the works."),
            Selection(reason="try damages", ref=refs[1]),
            Reading(sufficient=True, response="2% per week."),
        ]
    )
    trace = await NavigateStrategy(model, NavigateConfig(direct_char_threshold=0)).run(
        index_of(document_json), "late delivery penalty"
    )

    assert trace.status is RunStatus.ANSWERED
    assert trace.llm_calls == 4


# -- the frontier is actually parallel ----------------------------------------


async def test_candidates_are_read_concurrently() -> None:
    """Latency is one round-trip per round, not one per unit."""
    document_json = _contract_json()
    in_flight = 0
    peak = 0

    class ConcurrentModel(FakeChatModel):
        async def structured(self, messages, *, schema, max_tokens=None):  # type: ignore[no-untyped-def]
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            try:
                return await super().structured(messages, schema=schema, max_tokens=max_tokens)
            finally:
                in_flight -= 1

    model = ConcurrentModel(
        [
            Reading(sufficient=False, response="not here"),
            Reading(sufficient=False, response="not here either"),
            Reading(sufficient=False, response="nor here"),
            "Nothing conclusive.",
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3, max_steps=3, decisive_ratio=WIDE)
    await HybridStrategy(model, config).run(index_of(document_json), "works delivery invoices")

    assert peak == 3, f"expected three concurrent reads, saw {peak}"


async def test_steps_follow_retrieval_order_not_completion_order() -> None:
    """A trace that depends on which call returned first is not reproducible."""
    document_json = _contract_json()

    class JitteredModel(FakeChatModel):
        async def structured(self, messages, *, schema, max_tokens=None):  # type: ignore[no-untyped-def]
            # The first read launched sleeps longest, so completion order is
            # the reverse of launch order.
            await asyncio.sleep(0.003 * (3 - len(self.structured_calls)))
            return await super().structured(messages, schema=schema, max_tokens=max_tokens)

    model = JitteredModel(
        [
            Reading(sufficient=False, response="a"),
            Reading(sufficient=False, response="b"),
            Reading(sufficient=False, response="c"),
            "Nothing conclusive.",
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3, max_steps=3, decisive_ratio=WIDE)
    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "works delivery invoices"
    )

    assert [s.response for s in trace.steps] == ["a", "b", "c"]
    assert [s.index for s in trace.steps] == [1, 2, 3]


async def test_the_best_ranked_unit_wins_a_tie() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Reading(sufficient=True, response="from the top-ranked section"),
            Reading(sufficient=True, response="from the runner-up"),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=2, decisive_ratio=WIDE)
    trace = await HybridStrategy(model, config).run(index_of(document_json), "delivery invoices")

    assert trace.answer == "from the top-ranked section"
    assert len(trace.steps) == 2, "both reads are still recorded"


# -- falling back to the model ------------------------------------------------


async def test_a_paraphrased_question_falls_back_to_model_navigation() -> None:
    """Lexical retrieval misses paraphrase; that is why it never decides alone."""
    document_json = _contract_json()
    refs = _refs(document_json)
    model = FakeChatModel(
        [
            Selection(reason="damages is where sanctions live", ref=refs[1]),
            Reading(sufficient=True, response="2% per week."),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3)

    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "quelles sanctions en cas de retard"
    )

    assert trace.status is RunStatus.ANSWERED
    assert trace.steps[0].reason == "damages is where sanctions live"


async def test_a_model_fallback_that_misfires_is_recorded_as_such() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Selection(reason="nope", ref="#/texts/999"),
            Selection(reason="still nope", ref="#/texts/998"),
            Reading(sufficient=True, response="ok"),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3)
    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "zzz unrelated vocabulary"
    )

    assert trace.steps[0].fallback is True


# -- outcomes and budgets ------------------------------------------------------


async def test_two_absent_votes_end_the_run_honestly() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Reading(sufficient=False, response="not about cats", absent=True),
            Reading(sufficient=False, response="still not", absent=True),
            "This document does not discuss cats.",
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=2, decisive_ratio=WIDE)
    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "works delivery invoices"
    )

    assert trace.status is RunStatus.NOT_IN_DOCUMENT


async def test_the_batch_shrinks_to_the_remaining_budget() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Reading(sufficient=False, response="a"),
            Reading(sufficient=False, response="b"),
            "Nothing conclusive.",
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3, max_steps=2, decisive_ratio=WIDE)
    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "works delivery invoices"
    )

    assert len(trace.steps) == 2
    assert trace.status is RunStatus.BUDGET_EXHAUSTED


async def test_one_failing_read_does_not_lose_the_round() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            RuntimeError("backend hiccup"),
            Reading(sufficient=True, response="survived"),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=2, decisive_ratio=WIDE)
    trace = await HybridStrategy(model, config).run(index_of(document_json), "delivery invoices")

    assert trace.status is RunStatus.ANSWERED
    assert trace.answer == "survived"
    assert len(trace.steps) == 1


async def test_a_whole_round_failing_surfaces_the_error() -> None:
    document_json = _contract_json()
    model = FakeChatModel([RuntimeError("backend down"), RuntimeError("still down")])
    config = HybridConfig(direct_char_threshold=0, fanout=2, decisive_ratio=WIDE)

    try:
        await HybridStrategy(model, config).run(index_of(document_json), "delivery invoices")
    except RuntimeError as exc:
        assert "backend down" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected the backend error to propagate")


async def test_a_short_document_still_takes_the_cheap_path(flat_json: str) -> None:
    model = FakeChatModel([Reading(sufficient=True, response="12.4M EUR.")])
    trace = await HybridStrategy(model).run(index_of(flat_json), "revenue?")

    assert trace.llm_calls == 1
    assert len(trace.steps) == 1


async def test_an_empty_document_is_reported_not_answered() -> None:
    empty = DoclingDocument(name="empty").model_dump_json()
    trace = await HybridStrategy(FakeChatModel([])).run(index_of(empty), "q")
    assert trace.status is RunStatus.NOT_IN_DOCUMENT


# -- adaptive fanout and gap-driven re-query ----------------------------------


async def test_a_decisive_shortlist_is_read_alone() -> None:
    """Fanout buys breadth when the ranking is flat. When one section clearly
    wins, spending three calls to confirm it is waste."""
    document_json = _contract_json()
    model = FakeChatModel([Reading(sufficient=True, response="2% per week.")])
    config = HybridConfig(direct_char_threshold=0, fanout=3, decisive_ratio=1.0)

    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "late delivery penalty"
    )

    assert len(trace.steps) == 1
    assert trace.llm_calls == 1


async def test_a_flat_shortlist_is_read_broadly() -> None:
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Reading(sufficient=False, response="a"),
            Reading(sufficient=False, response="b"),
            Reading(sufficient=False, response="c"),
            "Nothing conclusive.",
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=3, max_steps=3, decisive_ratio=WIDE)

    trace = await HybridStrategy(model, config).run(
        index_of(document_json), "works delivery invoices"
    )

    assert len(trace.steps) == 3


async def test_the_second_round_searches_for_what_is_missing() -> None:
    """Upstream repeats the same query forever. The reader just said what it
    lacked; that is what the next lookup should be about."""
    document_json = _contract_json()
    model = FakeChatModel(
        [
            Reading(sufficient=False, response="Scope is silent; look for invoicing terms."),
            Reading(sufficient=True, response="Payable within 30 days."),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=1, decisive_ratio=WIDE)

    trace = await HybridStrategy(model, config).run(index_of(document_json), "works")

    assert trace.status is RunStatus.ANSWERED
    assert trace.steps[0].title == "Scope of works"
    assert trace.steps[1].title == "Invoicing", "the stated gap steered round two"


async def test_retrieval_stands_aside_once_every_query_term_has_been_read() -> None:
    """More term matching cannot help; the model's judgement can."""
    document_json = _contract_json()
    refs = _refs(document_json)
    model = FakeChatModel(
        [
            Reading(sufficient=False, response="seen it"),
            Selection(reason="structure suggests invoicing", ref=refs[2]),
            Reading(sufficient=True, response="30 days."),
        ]
    )
    config = HybridConfig(direct_char_threshold=0, fanout=1, decisive_ratio=WIDE, gap_notes=0)

    trace = await HybridStrategy(model, config).run(index_of(document_json), "liquidated damages")

    assert trace.steps[1].reason == "structure suggests invoicing"
