"""Pins the behaviour glosa mirrors from Studio's `infra/docling_tree.py`.

glosa cannot import that module, so parity is pinned by behaviour, taken from
its docstrings and from `infra/docling_graph.py`'s use of it. When Studio
changes a collapse rule, these are the tests that should go red.

They also stand as the acceptance criteria for injecting Studio's own
`DoclingTreeReader` through the `TreeReader` port: any implementation that
passes these is interchangeable with the bundled one.
"""

from __future__ import annotations

import json
from typing import Any

from glosa.studio.ports import TreeReader
from glosa.studio.tree import (
    LABEL_MAP,
    BundledTreeReader,
    build_collapse_index,
    dfs_order,
    element_label,
    is_section_header,
    iter_pages,
    iter_provs,
)
from tests.conftest import build_flat, build_inline, build_nested, build_picture


def _data(builder: Any) -> dict[str, Any]:
    return json.loads(builder().model_dump_json())


def test_the_bundled_reader_satisfies_the_port() -> None:
    assert isinstance(BundledTreeReader(), TreeReader)


def test_inline_style_runs_are_skipped_and_aggregated() -> None:
    """Studio #197: one `groups[]` entry plus N `texts[]` runs → one node."""
    doc_data = _data(build_inline)
    skip_refs, inline_meta = build_collapse_index(doc_data)

    assert {"#/texts/1", "#/texts/2", "#/texts/3"} <= skip_refs
    assert "#/groups/0" not in skip_refs
    meta = inline_meta["#/groups/0"]
    assert meta["text"] == "The fee is EUR 4,500 per month."
    assert [p["order"] for p in meta["provs"]] == [0, 1, 2], "prov order is re-indexed 0..N-1"


def test_picture_descendants_are_skipped_but_the_picture_stays() -> None:
    doc_data = _data(build_picture)
    skip_refs, _ = build_collapse_index(doc_data)

    picture_ref = doc_data["pictures"][0]["self_ref"]
    child_ref = doc_data["pictures"][0]["children"][0]["cref"]
    assert picture_ref not in skip_refs
    assert child_ref in skip_refs


def test_dfs_order_is_the_next_chain_and_omits_skipped_refs() -> None:
    doc_data = _data(build_inline)
    skip_refs, _ = build_collapse_index(doc_data)
    order = dfs_order(doc_data, skip_refs)

    assert order == ["#/texts/0", "#/groups/0"]
    assert not set(order) & skip_refs


def test_flat_and_nested_documents_share_a_reading_order() -> None:
    flat = dfs_order(_data(build_flat), set())
    nested = dfs_order(_data(build_nested), set())
    assert flat == nested


def test_title_and_section_header_both_project_to_sectionheader() -> None:
    """Which is why both are section boundaries in the UI."""
    assert element_label("title") == "SectionHeader"
    assert element_label("section_header") == "SectionHeader"
    assert is_section_header({"label": "title"})
    assert is_section_header({"label": "section_header"})
    assert not is_section_header({"label": "text"})


def test_unknown_labels_fall_back_rather_than_raise() -> None:
    assert element_label("something_new") == "TextElement"
    assert "list_item" in LABEL_MAP


def test_provs_are_flattened_with_order_page_and_charspan() -> None:
    doc_data = _data(build_flat)
    item = doc_data["texts"][0]
    rows = iter_provs(item)

    assert rows[0]["order"] == 0
    assert rows[0]["page_no"] == 1
    assert rows[0]["coord_origin"] == "BOTTOMLEFT"
    assert rows[0]["charspan_start"] == 0


def test_pages_are_yielded_with_their_size() -> None:
    pages = list(iter_pages(_data(build_flat)))
    assert [p["page_no"] for p in pages] == [1, 2]
    assert pages[0]["height"] == 792.0


def test_both_ref_spellings_are_accepted() -> None:
    """Studio accepts `$ref` and `cref` on child pointers; so must we."""
    doc_data = {
        "body": {"children": [{"$ref": "#/texts/0"}, {"cref": "#/texts/1"}]},
        "texts": [
            {"self_ref": "#/texts/0", "label": "text", "text": "a", "children": []},
            {"self_ref": "#/texts/1", "label": "text", "text": "b", "children": []},
        ],
    }
    assert dfs_order(doc_data, set()) == ["#/texts/0", "#/texts/1"]
