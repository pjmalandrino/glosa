"""The projection itself: collapses, rendering, truncation, malformed input."""

from __future__ import annotations

import json

import pytest

from glosa.domain.errors import DocumentParseError
from glosa.domain.index import TRUNCATION_MARKER
from glosa.infra.docling.projection import DoclingProjection, node_id_for
from tests.conftest import build_flat, index_of, pages, prov


def test_inline_group_is_one_node_not_its_style_runs(inline_json: str) -> None:
    """Studio issue #197. The style runs are in `skip_refs`; emitting one of
    their refs in a trace would highlight nothing in the graph."""
    projection = DoclingProjection.from_json(inline_json)
    refs = [e.self_ref for e in projection.elements]

    assert "#/groups/0" in refs
    assert not any(ref.startswith("#/texts/1") for ref in refs), "style runs must be collapsed"

    group = projection.by_ref["#/groups/0"]
    assert group.text == "The fee is EUR 4,500 per month."
    assert group.graph_label == "Paragraph"
    assert len(group.provs) == 3, "the group carries the union of its runs' provenances"


def test_the_collapsed_paragraph_is_what_gets_read(inline_json: str) -> None:
    index = index_of(inline_json)
    excerpt = index.excerpt(index.units[0].ref)

    assert "EUR 4,500" in excerpt.text
    assert node_id_for("#/groups/0") in excerpt.node_ids


def test_picture_children_are_dropped_but_the_caption_is_kept(picture_json: str) -> None:
    projection = DoclingProjection.from_json(picture_json)
    texts = " ".join(e.text for e in projection.elements)

    assert "NOISE-AXIS-LABEL" not in texts
    assert "deployment topology" in texts
    assert any(e.graph_label == "Figure" for e in projection.elements)


def test_tables_render_as_html_so_cells_stay_associated(table_json: str) -> None:
    """Markdown flattening loses row/column association — exactly what a
    numeric question needs. Rendered from the cell offsets, no docling-core."""
    index = index_of(table_json)
    text = index.excerpt(index.units[0].ref).text

    assert "<table>" in text
    assert "<th>Revenue</th>" in text
    assert "<td>12.4M</td>" in text


def test_graph_labels_match_studios_legend(flat_json: str) -> None:
    projection = DoclingProjection.from_json(flat_json)
    labels = {e.self_ref: e.graph_label for e in projection.elements}

    # `title` and `section_header` both project to SectionHeader — which is
    # what makes both of them section boundaries in the UI.
    assert labels["#/texts/0"] == "SectionHeader"
    assert labels["#/texts/1"] == "SectionHeader"
    assert labels["#/texts/2"] == "Paragraph"


def test_truncation_is_announced_not_silent() -> None:
    doc = build_flat()
    heading = doc.add_heading(text="Appendix", level=1, prov=prov(2, 500))
    doc.add_text(label="text", text="x" * 5_000, parent=heading, prov=prov(2, 480))

    index = index_of(doc.model_dump_json())
    appendix = next(u for u in index.units if u.title == "Appendix")
    excerpt = index.excerpt(appendix.ref, char_budget=1_000)

    assert excerpt.truncated is True
    assert TRUNCATION_MARKER.strip() in excerpt.text
    assert len(excerpt.text) < 2_000


def test_an_element_that_does_not_fit_is_dropped_whole() -> None:
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument(name="tiny-budget")
    pages(doc, 1)
    heading = doc.add_heading(text="Body", level=1, prov=prov(1, 740))
    doc.add_text(label="text", text="a" * 100, parent=heading, prov=prov(1, 700))
    doc.add_text(label="text", text="b" * 100, parent=heading, prov=prov(1, 680))

    index = index_of(doc.model_dump_json())
    excerpt = index.excerpt(index.units[0].ref, char_budget=120)

    assert "b" * 100 not in excerpt.text
    assert excerpt.truncated is True


def test_full_excerpt_covers_the_document_and_anchors_on_a_real_node(flat_json: str) -> None:
    index = index_of(flat_json)
    excerpt = index.full_excerpt()

    assert "12.4M EUR" in excerpt.text
    assert "supplier dispute" in excerpt.text
    assert node_id_for(excerpt.ref) in index.projection.node_ids
    assert set(excerpt.node_ids) <= index.projection.node_ids


def test_a_cycle_in_the_children_projects_each_node_once() -> None:
    """A corrupt `children` graph must terminate, not recurse forever."""
    payload = json.dumps(
        {
            "name": "cyclic",
            "body": {"children": [{"$ref": "#/texts/0"}]},
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "text",
                    "text": "a",
                    "children": [{"$ref": "#/texts/1"}],
                    "prov": [],
                },
                {
                    "self_ref": "#/texts/1",
                    "label": "text",
                    "text": "b",
                    "children": [{"$ref": "#/texts/0"}],
                    "prov": [],
                },
            ],
        }
    )
    projection = DoclingProjection.from_json(payload)

    assert [e.self_ref for e in projection.elements] == ["#/texts/0", "#/texts/1"]


def test_a_ref_listed_under_two_parents_is_projected_once() -> None:
    """Duplicating it doubled its text in every scope, excerpt and char_len."""
    payload = json.dumps(
        {
            "name": "dupes",
            "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "text",
                    "text": "once",
                    "children": [{"$ref": "#/texts/1"}],
                    "prov": [],
                },
                {"self_ref": "#/texts/1", "label": "text", "text": "twice?", "prov": []},
            ],
        }
    )
    projection = DoclingProjection.from_json(payload)

    assert [e.self_ref for e in projection.elements] == ["#/texts/0", "#/texts/1"]


@pytest.mark.parametrize(
    "payload",
    [
        # texts is not a list / items are not dicts
        '{"name": "x", "body": {"children": []}, "texts": "oops"}',
        '{"name": "x", "body": {"children": []}, "texts": [1, 2]}',
        # garbage bbox / charspan inside a prov row
        (
            '{"name": "x", "body": {"children": [{"$ref": "#/texts/0"}]}, "texts": '
            '[{"self_ref": "#/texts/0", "label": "text", "text": "t", '
            '"prov": [{"page_no": 1, "bbox": {"l": "abc"}, "charspan": ["x", "y"]}]}]}'
        ),
        # pages map holding non-dicts, or not a map at all
        '{"name": "x", "pages": {"1": "not-a-dict"}, "body": {"children": []}, "texts": []}',
        '{"name": "x", "pages": "zzz", "body": {"children": []}, "texts": []}',
        # a table whose grid metadata is garbage
        (
            '{"name": "x", "body": {"children": [{"$ref": "#/tables/0"}]}, "tables": '
            '[{"self_ref": "#/tables/0", "label": "table", "prov": [], "data": '
            '{"num_rows": "abc", "num_cols": null, "table_cells": '
            '[{"text": "c", "start_row_offset_idx": "?", "end_row_offset_idx": 1, '
            '"start_col_offset_idx": 0, "end_col_offset_idx": 1}]}}]}'
        ),
    ],
)
def test_garbage_payloads_degrade_or_raise_the_typed_error(payload: str) -> None:
    """The port's promise: a payload that is not a document either projects in
    a degraded form or raises `DocumentParseError` — never a raw
    RecursionError / AttributeError / ValueError."""
    import contextlib

    with contextlib.suppress(DocumentParseError):
        DoclingProjection.from_json(payload)


def test_orphan_nodes_stay_addressable() -> None:
    """An element unreachable from `body` is still a node in the projection, so
    it must still be indexed rather than silently disappearing."""
    payload = (
        '{"name": "orphans", "body": {"children": []}, '
        '"texts": [{"self_ref": "#/texts/0", "label": "text", "text": "stranded", '
        '"parent": {"cref": "#/body"}, "children": [], "prov": []}]}'
    )
    projection = DoclingProjection.from_json(payload)
    assert [e.self_ref for e in projection.elements] == ["#/texts/0"]
