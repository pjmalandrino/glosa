"""Excerpting: tables, truncation, page serialization."""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from docling_core.types.doc.base import Size

from glosa.document.excerpt import TRUNCATION_MARKER
from glosa.document.index import DocIndex


def _table_doc() -> DoclingDocument:
    doc = DoclingDocument(name="figures")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    heading = doc.add_heading(text="Financials", level=1)
    cells = [
        TableCell(
            text="Year",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            column_header=True,
        ),
        TableCell(
            text="Revenue",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            column_header=True,
        ),
        TableCell(
            text="2025",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
        TableCell(
            text="12.4M",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    ]
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells), parent=heading)
    return doc


def test_tables_are_serialized_as_html_so_cells_stay_associated() -> None:
    """Markdown flattening loses row/column association — exactly what a
    numeric question needs. Upstream only uses HTML in page mode."""
    index = DocIndex(_table_doc())
    text = index.excerpt(index.units[0].ref).text
    assert "<table" in text
    assert "12.4M" in text
    assert "Revenue" in text


def test_truncation_is_announced_not_silent() -> None:
    doc = DoclingDocument(name="long")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    heading = doc.add_heading(text="Body", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="x" * 5_000, parent=heading)

    index = DocIndex(doc)
    excerpt = index.excerpt(index.units[0].ref, char_budget=1_000)
    assert excerpt.truncated is True
    assert TRUNCATION_MARKER.strip() in excerpt.text
    assert len(excerpt.text) < 2_000


def test_short_items_are_dropped_whole_rather_than_cut_to_a_stub() -> None:
    doc = DoclingDocument(name="tiny-budget")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    heading = doc.add_heading(text="Body", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="a" * 100, parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="b" * 100, parent=heading)

    index = DocIndex(doc)
    excerpt = index.excerpt(index.units[0].ref, char_budget=120)
    assert "b" * 100 not in excerpt.text
    assert excerpt.truncated is True


def test_page_excerpt_covers_only_that_page(headless_doc: DoclingDocument) -> None:
    index = DocIndex(headless_doc)
    page_two = index.excerpt("#/pages/2")
    assert "Second page" in page_two.text
    assert "First page" not in page_two.text
    assert page_two.pages == (2,)


def test_full_excerpt_covers_the_whole_document(flat_doc: DoclingDocument) -> None:
    text = DocIndex(flat_doc).full_excerpt().text
    assert "12.4M EUR" in text
    assert "supplier dispute" in text
