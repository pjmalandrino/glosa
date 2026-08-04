"""Outline rendering under a character budget."""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.base import Size

from glosa.document.index import DocIndex
from glosa.document.outline import render_outline


def _big_doc(sections: int, depth_two_per_section: int = 3) -> DoclingDocument:
    doc = DoclingDocument(name="big")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    for i in range(sections):
        heading = doc.add_heading(text=f"Chapter {i} on regulatory reporting", level=1)
        doc.add_text(label=DocItemLabel.TEXT, text=f"Body of chapter {i}. " * 20, parent=heading)
        for j in range(depth_two_per_section):
            sub = doc.add_heading(
                text=f"Section {i}.{j} detailed provisions", level=2, parent=heading
            )
            doc.add_text(label=DocItemLabel.TEXT, text=f"Detail {i}.{j}. " * 20, parent=sub)
    return doc


def test_every_unit_is_listed_with_its_ref(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    text = render_outline(index.units)
    for unit in index.units:
        assert unit.ref in text
        assert unit.title in text


def test_visited_units_are_marked(flat_doc: DoclingDocument) -> None:
    index = DocIndex(flat_doc)
    target = index.units[1].ref
    line = next(
        line
        for line in render_outline(index.units, visited=[target]).splitlines()
        if target in line
    )
    assert "✓" in line


def test_deepest_levels_are_dropped_before_anything_is_elided() -> None:
    index = DocIndex(_big_doc(sections=30))
    text = render_outline(index.units, char_budget=2_000)
    assert len(text) <= 2_000
    assert "Chapter 0" in text
    # h2 entries are the first thing to go.
    assert "Section 0.0" not in text


def test_elision_is_announced_with_a_count() -> None:
    """A silently shortened map reads as a complete one. It must not."""
    index = DocIndex(_big_doc(sections=200, depth_two_per_section=0))
    text = render_outline(index.units, char_budget=1_500)
    assert len(text) <= 1_500
    assert "omitted from this map" in text
    assert "Chapter 0" in text
    assert "Chapter 199" in text


def test_empty_document_renders_a_placeholder() -> None:
    assert render_outline([]) == "(empty document)"
