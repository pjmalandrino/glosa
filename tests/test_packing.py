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
    """One of the two signals that hand the round to the model."""
    index = index_of(_sections_json())
    shortlist = index.ranker.shortlist("cryptocurrency custody")
    assert not shortlist
    assert shortlist.confident(min_coverage=0.0, min_margin=0.0) is False


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


def test_an_oversized_matching_passage_is_truncated_in_not_dropped() -> None:
    """The worst case used to be the motivating one: the answer inside a
    single paragraph larger than the whole budget. Skipping it while
    zero-score filler filled the excerpt read as diligence and was blindness."""
    doc = DoclingDocument(name="appendix")
    pages(doc, 1)
    heading = doc.add_heading(text="Appendix C", level=1, prov=prov(1, 740))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="The indemnity cap is set at 500,000 EUR. "
        + "Further conditions apply to that cap under schedule 4. " * 60,
        parent=heading,
        prov=prov(1, 700),
    )
    for i in range(3):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Unrelated administrative paragraph number {i}. " * 8,
            parent=heading,
            prov=prov(1, 650 - i),
        )
    index = index_of(doc.model_dump_json())

    excerpt = index.excerpt(index.units[0].ref, char_budget=600, focus="what is the indemnity cap")

    assert "500,000 EUR" in excerpt.text
    assert excerpt.truncated is True


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


# --- how much to trust a shortlist --------------------------------------------


def _french_contract_json() -> str:
    """The case that motivated the confidence gate.

    "montant" and "fournisseur" appear in Article 1 and nowhere near the answer,
    which lives in Article 7 under words the question never uses.
    """
    doc = DoclingDocument(name="marche")
    pages(doc, 1)
    for title, body in [
        (
            "Article 1 — Objet du marche",
            "Le present marche a pour objet la fourniture de prestations par le "
            "fournisseur. Le montant global du marche est fixe a l'acte d'engagement. " * 12,
        ),
        (
            "Article 7 — Plafond d'indemnisation",
            "La responsabilite du titulaire est limitee a 500 000 euros par sinistre. " * 12,
        ),
        (
            "Article 9 — Delais de reglement",
            "Le paiement intervient dans les trente jours suivant reception. " * 12,
        ),
    ]:
        heading = doc.add_heading(text=title, level=1, prov=prov(1, 740))
        doc.add_text(label=DocItemLabel.TEXT, text=body, parent=heading, prov=prov(1, 700))
    return doc.model_dump_json()


def test_a_matching_question_produces_a_confident_shortlist() -> None:
    index = index_of(_french_contract_json())
    shortlist = index.ranker.shortlist("quel est le plafond d'indemnisation ?")

    assert shortlist.top is not None
    assert "Plafond" in shortlist.top.unit.title
    assert shortlist.margin > 1.5
    assert shortlist.confident(min_coverage=0.34, min_margin=1.15) is True


def test_a_spurious_match_produces_a_non_empty_but_untrusted_shortlist() -> None:
    """This is the bug the gate closes: the list is not empty, so the old
    `if not candidates` test kept the model out — and the wrong section was
    read with a confident-sounding rationale."""
    index = index_of(_french_contract_json())
    shortlist = index.ranker.shortlist(
        "quel est le montant maximum que le fournisseur devra rembourser ?"
    )

    assert shortlist, "a spurious match still fills the list"
    assert "Objet" in shortlist.candidates[0].unit.title, "and it is the wrong section"
    assert shortlist.margin < 1.15, "but the ranking is flat — that is the tell"
    assert shortlist.confident(min_coverage=0.34, min_margin=1.15) is False


def test_an_expansion_rescues_the_paraphrase() -> None:
    """Vocabulary the model supplies, fused as extra views."""
    index = index_of(_french_contract_json())
    question = "quel est le montant maximum que le fournisseur devra rembourser ?"
    rescued = index.ranker.shortlist(
        question, expansion=["plafond", "indemnisation", "responsabilite", "sinistre"]
    )

    assert rescued.top is not None
    assert "Plafond" in rescued.top.unit.title
    assert rescued.confident(min_coverage=0.34, min_margin=1.15) is True
    assert any(name.endswith("+") for name, _ in rescued.top.ranks)


def test_an_expansion_cannot_displace_what_the_users_own_words_found() -> None:
    """A bad expansion may add candidates; it must not overrule a good match."""
    index = index_of(_french_contract_json())
    question = "quel est le plafond d'indemnisation ?"
    plain = index.ranker.shortlist(question)
    noisy = index.ranker.shortlist(question, expansion=["reglement", "paiement", "jours"])

    assert plain.top is not None and noisy.top is not None
    assert noisy.top.unit.ref == plain.top.unit.ref


def test_coverage_ignores_words_the_document_never_uses() -> None:
    """Counting them would punish every candidate equally and measure nothing."""
    index = index_of(_french_contract_json())
    body = index.ranker.body
    ref = index.units[1].ref

    assert body.coverage("plafond indemnisation cryptomonnaie", [ref]) == 1.0
    assert body.coverage("plafond reglement", [ref]) == 0.5


def test_the_expansion_weight_stays_inside_its_guarantee() -> None:
    """The property is arithmetic, so pin it: all three expansion views together
    must contribute less than one original view at rank 1."""
    from glosa.domain.rank import EXPANSION_WEIGHT, RRF_K, SUMMARY_WEIGHT, UNIT_SCORE

    expansion_ceiling = (1.0 + 1.0 + SUMMARY_WEIGHT) * EXPANSION_WEIGHT / (RRF_K + 1)
    assert expansion_ceiling < UNIT_SCORE
