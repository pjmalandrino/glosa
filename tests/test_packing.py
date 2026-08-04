"""Ranking units, and packing an over-budget section around the question."""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument

from glosa.domain.index import SELECTION_MARKER
from tests.conftest import index_of, pages, prov


def _long_section_json() -> str:
    """One section whose answer sits at the very end — the head-first trap."""
    doc = DoclingDocument(name="appendix")
    pages(doc, 1)
    heading = doc.add_heading(text="Appendix C", level=1, prov=prov(1, 740))
    for i in range(60):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Paragraph {i} about unrelated administrative arrangements. " * 6,
            parent=heading,
            prov=prov(1, 700 - i),
        )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="The indemnity cap is set at 500,000 EUR.",
        parent=heading,
        prov=prov(1, 600),
    )
    return doc.model_dump_json()


def _sections_json() -> str:
    doc = DoclingDocument(name="contract")
    pages(doc, 1)
    for title, body in [
        ("Article 1", "The supplier shall deliver the works described in Annex A."),
        ("Article 2", "Late delivery incurs a penalty of 2% per week."),
        ("Indemnity", "Nothing further is specified in this clause."),
    ]:
        heading = doc.add_heading(text=title, level=1, prov=prov(1, 740))
        doc.add_text(label=DocItemLabel.TEXT, text=body, parent=heading, prov=prov(1, 700))
    return doc.model_dump_json()


# --- ranking ------------------------------------------------------------------


def test_the_body_finds_what_an_uninformative_heading_hides() -> None:
    """ "Article 2" says nothing; its text says everything."""
    index = index_of(_sections_json())
    best = index.ranker.shortlist("late delivery penalty")[0]
    assert best.unit.title == "Article 2"


def test_a_matching_heading_counts_even_with_a_thin_body() -> None:
    index = index_of(_sections_json())
    refs = [c.unit.title for c in index.ranker.shortlist("indemnity")]
    assert refs[0] == "Indemnity"


def test_the_rationale_says_where_the_match_came_from() -> None:
    index = index_of(_sections_json())
    best = index.ranker.shortlist("indemnity")[0]
    assert "heading rank 1" in best.rationale


def test_visited_units_are_excluded_from_the_shortlist() -> None:
    index = index_of(_sections_json())
    first = index.ranker.shortlist("delivery")[0]
    again = index.ranker.shortlist("delivery", exclude=[first.ref])
    assert first.ref not in [c.ref for c in again]


def test_no_lexical_signal_yields_an_empty_shortlist() -> None:
    """The signal the loop uses to hand the round to the model."""
    index = index_of(_sections_json())
    assert index.ranker.shortlist("cryptocurrency custody") == []


def test_the_ranker_is_built_once_per_document() -> None:
    index = index_of(_sections_json())
    assert index.ranker is index.ranker


# --- query-aware packing ------------------------------------------------------


def test_an_over_budget_section_keeps_the_passages_that_match() -> None:
    """Head-first truncation would drop the one paragraph that answers."""
    index = index_of(_long_section_json())
    ref = index.units[0].ref

    blind = index.excerpt(ref, char_budget=600)
    focused = index.excerpt(ref, char_budget=600, focus="what is the indemnity cap")

    assert "indemnity cap" not in blind.text
    assert "500,000 EUR" in focused.text


def test_the_kept_passages_stay_in_reading_order() -> None:
    index = index_of(_long_section_json())
    ref = index.units[0].ref
    excerpt = index.excerpt(ref, char_budget=900, focus="indemnity cap paragraph 3")

    orders = [p.self_ref for p in excerpt.parts]
    assert orders == sorted(orders, key=lambda r: int(r.rsplit("/", 1)[1]))


def test_omission_is_announced_with_a_count() -> None:
    index = index_of(_long_section_json())
    excerpt = index.excerpt(index.units[0].ref, char_budget=600, focus="indemnity cap")

    assert excerpt.truncated is True
    assert "omitted" in excerpt.text
    assert SELECTION_MARKER.split("{n}")[1].strip() in excerpt.text


def test_a_section_that_fits_is_untouched_by_focus() -> None:
    index = index_of(_sections_json())
    ref = index.units[0].ref
    assert index.excerpt(ref).text == index.excerpt(ref, focus="anything at all").text


def test_focus_that_matches_nothing_falls_back_to_head_first() -> None:
    index = index_of(_long_section_json())
    ref = index.units[0].ref
    focused = index.excerpt(ref, char_budget=600, focus="zzz nonexistent vocabulary")

    assert focused.text.startswith("Appendix C")
    assert focused.truncated is True


def test_excerpts_are_cached_per_focus() -> None:
    index = index_of(_long_section_json())
    ref = index.units[0].ref
    assert index.excerpt(ref, focus="cap") is index.excerpt(ref, focus="cap")
    assert index.excerpt(ref, focus="cap") is not index.excerpt(ref, focus="other")


def test_retrieval_indexes_the_whole_section_not_the_budgeted_excerpt() -> None:
    """The bug this guards: indexing `excerpt()` capped the corpus at the
    default budget, so a term past that offset scored zero and the section was
    never shortlisted — in exactly the long sections query-aware packing exists
    to rescue."""
    index = index_of(_long_section_json())
    unit = index.units[0]

    assert unit.char_len > len(index.excerpt(unit.ref).text), "fixture must overflow the budget"
    assert index.ranker.shortlist("indemnity cap")[0].ref == unit.ref


# --- host enrichment as a retrieval signal ------------------------------------


def _enriched_json() -> str:
    """Sections whose headings say nothing and whose summaries say everything.

    This is the shape PageIndex builds its index around, and the shape
    `docling-agent`'s enricher writes into `meta.summary`.
    """
    from docling_core.types.doc.common.meta import BaseMeta, KeywordsMetaField, SummaryMetaField

    doc = DoclingDocument(name="contract")
    pages(doc, 1)
    written = [
        (
            "Article 11",
            "Aucune stipulation contraire.",
            "Governs the delivery timetable.",
            ["timetable"],
        ),
        (
            "Article 12",
            "Aucune stipulation contraire.",
            "Caps the supplier's liability.",
            ["liability cap"],
        ),
    ]
    for title, body, summary, keywords in written:
        heading = doc.add_heading(text=title, level=1, prov=prov(1, 740))
        heading.meta = BaseMeta(
            summary=SummaryMetaField(text=summary),
            keywords=KeywordsMetaField(values=keywords),
        )
        doc.add_text(label=DocItemLabel.TEXT, text=body, parent=heading, prov=prov(1, 700))
    return doc.model_dump_json()


def test_enrichment_is_read_off_the_document_when_the_host_produced_it() -> None:
    index = index_of(_enriched_json())
    unit = index.units[1]

    assert unit.summary == "Caps the supplier's liability."
    assert unit.keywords == ("liability cap",)
    assert unit.enriched is True
    assert index.ranker.uses_enrichment is True


def test_a_summary_finds_what_neither_heading_nor_body_contains() -> None:
    """ "Article 12" and "Aucune stipulation contraire" share no word with the
    question. Without the summary this shortlist is empty."""
    index = index_of(_enriched_json())
    best = index.ranker.shortlist("liability")

    assert best, "the summary is the only view that can match here"
    assert best[0].unit.title == "Article 12"
    assert "summary rank 1" in best[0].rationale


def test_a_document_without_enrichment_still_ranks_on_text() -> None:
    index = index_of(_sections_json())
    assert index.ranker.uses_enrichment is False
    assert index.ranker.shortlist("late delivery penalty")[0].unit.title == "Article 2"


def test_the_outline_shows_the_summary_so_the_model_can_navigate_by_it() -> None:
    from glosa.domain.outline import render_outline

    index = index_of(_enriched_json())
    text = render_outline(index.units).text

    assert "Caps the supplier's liability." in text
    assert "↳" in text


def test_the_outline_reports_which_refs_it_actually_showed() -> None:
    """Offering a model refs it cannot see invites it to name one at random."""
    from glosa.domain.outline import render_outline

    index = index_of(_sections_json())
    full = render_outline(index.units)
    assert full.refs == {u.ref for u in index.units}
    assert full.complete is True
