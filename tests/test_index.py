"""Structural indexing."""

from __future__ import annotations

import pytest
from docling_core.types.doc import DoclingDocument

from glosa.document.index import DocIndex
from glosa.errors import DocumentParseError
from glosa.types import UnitKind


def _units(doc: DoclingDocument) -> list[tuple[str, int, str]]:
    index = DocIndex(doc)
    return [(u.title, u.level, u.ref) for u in index.units]


def test_flat_and_nested_documents_index_identically(
    flat_doc: DoclingDocument, nested_doc: DoclingDocument
) -> None:
    """The whole point of the level stack.

    Upstream reads heading level off `iterate_items` tree depth, which is
    constant on a flat document, so its section scan degenerates. Both shapes
    must produce the same four units here.
    """
    assert _units(flat_doc) == _units(nested_doc)
    assert [title for title, _, _ in _units(flat_doc)] == [
        "Annual Report 2025",
        "Revenue",
        "Risks",
        "Legal",
    ]
    assert [level for _, level, _ in _units(flat_doc)] == [0, 1, 1, 2]


def test_subsection_is_nested_under_its_parent(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    risks = next(u for u in index.units if u.title == "Risks")
    legal = next(u for u in index.units if u.title == "Legal")
    assert risks.child_refs == (legal.ref,)


def test_excerpt_includes_subsections(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    risks = next(u for u in index.units if u.title == "Risks")
    text = index.excerpt(risks.ref).text
    assert "Legal" in text
    assert "supplier dispute" in text
    # A sibling section must not leak in.
    assert "Revenue reached" not in text


def test_excerpt_of_a_leaf_section_is_scoped(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    revenue = next(u for u in index.units if u.title == "Revenue")
    text = index.excerpt(revenue.ref).text
    assert "12.4M EUR" in text
    assert "supplier dispute" not in text


def test_headless_document_falls_back_to_pages(headless_doc: DoclingDocument) -> None:
    index = DocIndex(headless_doc)
    assert index.kind is UnitKind.PAGE
    assert [u.ref for u in index.units] == ["#/pages/1", "#/pages/2"]
    assert "First page" in index.excerpt("#/pages/1").text
    assert "Second page" not in index.excerpt("#/pages/1").text


def test_preamble_becomes_its_own_unit_anchored_on_a_real_ref(
    preamble_doc: DoclingDocument,
) -> None:
    index = DocIndex(preamble_doc)
    first = index.units[0]
    assert "Internal memo" in first.title
    # The ref must resolve in the caller's own projection of the tree.
    assert first.ref.startswith("#/texts/")
    assert "Internal memo" in index.excerpt(first.ref).text


def test_pages_and_bboxes_are_carried_on_excerpt_parts(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    revenue = next(u for u in index.units if u.title == "Revenue")
    excerpt = index.excerpt(revenue.ref)
    assert excerpt.pages == (1,)
    located = [p for p in excerpt.parts if p.bbox is not None]
    assert located, "expected at least one part with a normalized bbox"
    left, top, right, bottom = located[0].bbox or (0, 0, 0, 0)
    assert left < right and top < bottom, "bbox must be TOPLEFT with positive area"


def test_char_len_counts_the_whole_subtree(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    risks = next(u for u in index.units if u.title == "Risks")
    legal = next(u for u in index.units if u.title == "Legal")
    assert risks.char_len > legal.char_len


def test_excerpt_is_cached_per_budget(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    ref = index.units[1].ref
    assert index.excerpt(ref) is index.excerpt(ref)
    assert index.excerpt(ref, char_budget=50) is not index.excerpt(ref)


def test_unknown_ref_yields_an_empty_excerpt(flat_doc: DoclingDocument) -> None:
    assert DocIndex(flat_doc).excerpt("#/texts/999").text == ""


def test_from_json_round_trip(flat_doc: DoclingDocument) -> None:
    index = DocIndex.from_json(flat_doc.model_dump_json())
    assert len(index.units) == 4
    assert len(index.doc_hash) == 64


def test_from_json_rejects_garbage() -> None:
    with pytest.raises(DocumentParseError):
        DocIndex.from_json("{'not': 'a document'}")
