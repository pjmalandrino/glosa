"""Outline rendering under a character budget."""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument

from glosa.domain.outline import render_outline
from tests.conftest import index_of, pages, prov


def _big_doc_json(sections: int, depth_two_per_section: int = 3) -> str:
    doc = DoclingDocument(name="big")
    pages(doc, 1)
    for i in range(sections):
        heading = doc.add_heading(
            text=f"Chapter {i} on regulatory reporting", level=1, prov=prov(1, 740)
        )
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Body of chapter {i}. " * 20,
            parent=heading,
            prov=prov(1, 700),
        )
        for j in range(depth_two_per_section):
            sub = doc.add_heading(
                text=f"Section {i}.{j} detailed provisions", level=2, parent=heading
            )
            doc.add_text(label=DocItemLabel.TEXT, text=f"Detail {i}.{j}. " * 20, parent=sub)
    return doc.model_dump_json()


def test_every_unit_is_listed_with_its_ref(flat_json: str) -> None:
    index = index_of(flat_json)
    text = render_outline(index.units).text
    for unit in index.units:
        assert unit.ref in text
        assert unit.title in text


def test_visited_units_are_marked(flat_json: str) -> None:
    index = index_of(flat_json)
    target = index.units[1].ref
    line = next(
        line
        for line in render_outline(index.units, visited=[target]).text.splitlines()
        if target in line
    )
    assert "✓" in line


def test_deepest_levels_are_dropped_before_anything_is_elided() -> None:
    index = index_of(_big_doc_json(sections=30))
    text = render_outline(index.units, char_budget=2_000).text
    assert len(text) <= 2_000
    assert "Chapter 0" in text
    # h2 entries are the first thing to go.
    assert "Section 0.0" not in text


def test_elision_is_announced_with_a_count() -> None:
    """A silently shortened map reads as a complete one. It must not."""
    index = index_of(_big_doc_json(sections=200, depth_two_per_section=0))
    text = render_outline(index.units, char_budget=1_500).text
    assert len(text) <= 1_500
    assert "omitted from this map" in text
    assert "Chapter 0" in text
    assert "Chapter 199" in text


def test_empty_document_renders_a_placeholder() -> None:
    assert render_outline([]).text == "(empty document)"


def test_an_elided_map_reports_only_the_refs_it_showed() -> None:
    """The candidate list must shrink to the map, or the model is invited to
    name a section it never saw."""
    index = index_of(_big_doc_json(sections=200, depth_two_per_section=0))
    outline = render_outline(index.units, char_budget=1_500)

    assert outline.complete is False
    assert 0 < len(outline.refs) < len(index.units)
    assert all(ref in outline.text for ref in outline.refs)


def test_dropping_deep_levels_is_reported_as_incomplete() -> None:
    index = index_of(_big_doc_json(sections=30))
    outline = render_outline(index.units, char_budget=2_000)

    assert outline.complete is False
    assert all(u.level == 0 or u.level == 1 for u in index.units if u.ref in outline.refs)


def test_a_map_that_fits_is_complete() -> None:
    index = index_of(_big_doc_json(sections=2, depth_two_per_section=1))
    outline = render_outline(index.units)

    assert outline.complete is True
    assert outline.refs == {u.ref for u in index.units}


# -- leads ---------------------------------------------------------------------


def _numbered_json(sections: int) -> str:
    """Headings that name nothing — the case the lead exists for."""
    doc = DoclingDocument(name="marche")
    pages(doc, 1)
    for i in range(sections):
        heading = doc.add_heading(text=f"Article {i + 1}", level=1, prov=prov(1, 740))
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=(
                f"Cette clause traite du sujet numero {i} et de ses consequences "
                f"pour la partie concernee, dossier {i}. "
            )
            * 3,
            parent=heading,
            prov=prov(1, 700),
        )
    return doc.model_dump_json()


def test_a_numbered_heading_is_described_by_its_own_first_line() -> None:
    index = index_of(_numbered_json(4))
    text = render_outline(index.units).text

    assert "Article 1" in text
    assert "Cette clause traite du sujet numero 0" in text


def test_leads_are_dropped_before_sections_are() -> None:
    """Describing 37 of the sections is worse than listing all 100: the
    candidate list is restricted to the refs the map showed."""
    index = index_of(_numbered_json(100))
    outline = render_outline(index.units, char_budget=6_000)

    assert len(outline.refs) == len(index.units), "every section still listed"
    assert "↳" not in outline.text, "and the leads paid for it"
    # Non-vacuous: the same units do carry leads, the budget is what dropped them.
    assert render_outline(index.units, char_budget=20_000).text.count("↳") == 100


def test_leads_survive_a_budget_that_can_afford_them() -> None:
    index = index_of(_numbered_json(10))
    outline = render_outline(index.units, char_budget=6_000)

    assert len(outline.refs) == 10
    assert outline.text.count("↳") == 10


def test_a_visited_section_loses_its_lead() -> None:
    """Its own note is already in the prompt; the opening line repeats it."""
    index = index_of(_numbered_json(4))
    first = index.units[0].ref
    text = render_outline(index.units, visited=[first]).text

    assert "Cette clause traite du sujet numero 0" not in text
    assert "Cette clause traite du sujet numero 1" in text
