"""Retrieval units over Studio's projection."""

from __future__ import annotations

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData

from glosa.domain.errors import DocumentParseError
from glosa.domain.index import DocIndex
from glosa.domain.values import Element, UnitKind
from glosa.infra.docling.projection import node_id_for, page_node_id
from glosa.ports.document import DocumentProjection
from tests.conftest import FakeProjection, index_of, pages, prov


def _titles(document_json: str) -> list[str]:
    return [u.title for u in index_of(document_json).units]


def test_flat_and_nested_documents_produce_the_same_units(flat_json: str, nested_json: str) -> None:
    """Both shapes yield the same DFS reading order, so the same scopes.

    This is what makes a trace portable between a well-formed document and the
    flat output most PDF conversions produce.
    """
    flat = [(u.title, u.ref, u.level) for u in index_of(flat_json).units]
    nested = [(u.title, u.ref, u.level) for u in index_of(nested_json).units]
    assert flat == nested
    assert [t for t, _, _ in flat] == ["Annual Report 2025", "Revenue", "Risks", "Legal"]


def test_a_section_stops_at_the_next_heading_like_the_ui_does(flat_json: str) -> None:
    """`computeSectionParents` scopes a section from one SectionHeader to the
    next along the NEXT chain, with no level nesting. If glosa nested `Legal`
    inside `Risks`, a step claiming to read `Risks` would highlight a different
    set of nodes than it actually read."""
    index = index_of(flat_json)
    risks = next(u for u in index.units if u.title == "Risks")
    legal = next(u for u in index.units if u.title == "Legal")

    assert index.excerpt(risks.ref).text.strip() == "Risks"
    assert "supplier dispute" in index.excerpt(legal.ref).text
    assert "supplier dispute" not in index.excerpt(risks.ref).text


def test_every_unit_anchors_on_a_node_the_graph_has(flat_json: str) -> None:
    index = index_of(flat_json)
    projected = index.projection.node_ids
    for unit in index.units:
        assert unit.node_id in projected
        assert unit.node_id == node_id_for(unit.ref)


def test_excerpt_parts_carry_graph_node_ids(flat_json: str) -> None:
    index = index_of(flat_json)
    revenue = next(u for u in index.units if u.title == "Revenue")
    excerpt = index.excerpt(revenue.ref)

    assert excerpt.node_ids
    assert set(excerpt.node_ids) <= index.projection.node_ids


def test_headless_document_falls_back_to_pages(headless_json: str) -> None:
    index = index_of(headless_json)
    assert index.kind is UnitKind.PAGE
    assert [u.ref for u in index.units] == ["#/pages/1", "#/pages/2"]
    assert [u.node_id for u in index.units] == [page_node_id(1), page_node_id(2)]
    assert "First page" in index.excerpt("#/pages/1").text
    assert "Second page" not in index.excerpt("#/pages/1").text


def test_preamble_is_anchored_on_its_own_first_element(preamble_json: str) -> None:
    index = index_of(preamble_json)
    first = index.units[0]
    assert "Internal memo" in first.title
    assert first.node_id in index.projection.node_ids
    assert "Internal memo" in index.excerpt(first.ref).text


def test_provenance_is_normalized_to_topleft(flat_json: str) -> None:
    """Fixtures use BOTTOMLEFT, which is what Docling emits for PDFs."""
    index = index_of(flat_json)
    revenue = next(u for u in index.units if u.title == "Revenue")
    excerpt = index.excerpt(revenue.ref)

    assert excerpt.pages == (1,)
    located = [p for p in excerpt.parts if p.bbox is not None]
    assert located
    left, top, right, bottom = located[0].bbox or (0.0, 0.0, 0.0, 0.0)
    assert left < right and top < bottom


def test_furniture_is_addressable_but_not_read(furniture_json: str) -> None:
    index = index_of(furniture_json)
    header_node = node_id_for("#/texts/0")

    assert header_node in index.projection.node_ids, "must stay a node in the graph"
    assert all("CONFIDENTIAL" not in index.excerpt(u.ref).text for u in index.units)

    with_furniture = index_of(furniture_json, include_furniture=True)
    assert any("CONFIDENTIAL" in with_furniture.excerpt(u.ref).text for u in with_furniture.units)


def test_char_len_counts_the_content_of_the_scope(flat_json: str) -> None:
    """Used to size the outline, so it measures content, not separators."""
    index = index_of(flat_json)
    revenue = next(u for u in index.units if u.title == "Revenue")
    excerpt = index.excerpt(revenue.ref)
    assert revenue.char_len == sum(len(p.text) for p in excerpt.parts)


def test_excerpt_is_cached_per_budget(flat_json: str) -> None:
    index = index_of(flat_json)
    ref = index.units[1].ref
    assert index.excerpt(ref) is index.excerpt(ref)
    assert index.excerpt(ref, char_budget=10) is not index.excerpt(ref)


def test_unknown_ref_yields_an_empty_excerpt(flat_json: str) -> None:
    assert index_of(flat_json).excerpt("#/texts/999").text == ""


def test_the_index_runs_on_any_projection_not_just_doclings() -> None:
    """The inversion, demonstrated: a hand-written projection, no Docling."""
    a = Element(self_ref="a", node_id="n::a", text="Chapter", order=0, is_section=True)
    b = Element(self_ref="b", node_id="n::b", text="Body text.", order=1, pages=(7,), page_no=7)
    index = DocIndex(FakeProjection((a, b)))

    assert isinstance(index.projection, DocumentProjection)
    assert [u.ref for u in index.units] == ["a"]
    assert index.excerpt("a").text == "Chapter\n\nBody text."
    assert index.excerpt("a").node_ids == ("n::a", "n::b")


# -- the lead: a section's own opening line ------------------------------------


def _articles(*bodies: str, table_first: bool = False) -> str:
    """Numbered headings — the case where a table of contents says nothing."""
    doc = DoclingDocument(name="marche")
    pages(doc, 1)
    for i, body in enumerate(bodies):
        heading = doc.add_heading(text=f"Article {i + 1}", level=1, prov=prov(1, 740))
        if table_first and i == 0:
            cell = TableCell(
                text="Annee",
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                column_header=True,
            )
            doc.add_table(
                data=TableData(num_rows=1, num_cols=1, table_cells=[cell]),
                parent=heading,
                prov=prov(1, 720),
            )
        doc.add_text(label=DocItemLabel.TEXT, text=body, parent=heading, prov=prov(1, 700))
    return doc.model_dump_json()


def _leads(document_json: str) -> list[str]:
    return [u.lead for u in index_of(document_json).units]


def test_a_section_carries_its_opening_sentence() -> None:
    """What "Article 2" hides and the first line says for free."""
    leads = _leads(
        _articles(
            "Le present marche a pour objet la fourniture de prestations de nettoyage.",
            "La responsabilite du titulaire est limitee a 500 000 euros par sinistre.",
        )
    )

    assert leads[1].startswith("La responsabilite du titulaire est limitee a 500 000 euros")


def test_the_lead_stops_on_a_sentence_boundary() -> None:
    leads = _leads(
        _articles(
            "Le marche porte sur le nettoyage. Il court sur trois ans. "
            "Le titulaire est designe a l'acte d'engagement et ne peut ceder ses "
            "droits sans accord ecrit prealable du pouvoir adjudicateur.",
            "Les litiges relevent du tribunal administratif de Nancy.",
        )
    )

    assert leads[0] == "Le marche porte sur le nettoyage. Il court sur trois ans."


def test_a_boilerplate_opening_is_dropped_rather_than_shown() -> None:
    """Five sections opening the same way describe nothing and cost budget.
    Saying nothing is better than saying it five times."""
    same = "Le present article a pour objet de definir les conditions applicables aux parties."

    assert _leads(_articles(*[same] * 5)) == [""] * 5


def test_an_opening_that_only_restates_its_heading_is_dropped() -> None:
    """The heading is already printed on the line above."""
    leads = _leads(
        _articles(
            "Article 1.",
            "Les litiges relevent du tribunal administratif de Nancy.",
        )
    )

    assert leads[0] == ""
    assert leads[1].startswith("Les litiges")


def test_a_section_opening_on_a_table_leads_with_its_prose() -> None:
    """A serialized table quoted as an opening line reads as `<table><tr>…`."""
    leads = _leads(
        _articles(
            "Le bareme figure ci-dessus et s'applique a compter du 1er janvier.",
            "Les litiges relevent du tribunal administratif de Nancy.",
            table_first=True,
        )
    )

    assert leads[0] == "Le bareme figure ci-dessus et s'applique a compter du 1er janvier."


def test_pages_get_no_lead(headless_json: str) -> None:
    """A page unit's title is already its opening text."""
    index = index_of(headless_json)

    assert index.kind is UnitKind.PAGE
    assert all(u.lead == "" for u in index.units)


@pytest.mark.parametrize("payload", ["not json at all", "[]", '"a string"'])
def test_from_json_rejects_anything_that_is_not_a_document(payload: str) -> None:
    with pytest.raises(DocumentParseError):
        index_of(payload)


def test_an_empty_document_has_no_units() -> None:
    assert index_of('{"name": "empty", "body": {"children": []}}').units == ()
