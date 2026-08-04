"""Provenance helpers — `DocItem` → page number and TOPLEFT bounding box.

Docling emits bounding boxes in either TOPLEFT or BOTTOMLEFT origin. Every
bbox that leaves glosa is normalized to TOPLEFT `[l, t, r, b]`, which is what
Docling Studio's canvas overlay consumes. Degenerate boxes are dropped rather
than emitted as a zero-area rectangle, so a consumer can distinguish "no
location known" from "located at nothing".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from glosa.types import BBox, Span

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import DocItem


def page_height(doc: DoclingDocument, page_no: int) -> float | None:
    """Height of `page_no`, or None when the page is unknown to the document."""
    page = doc.pages.get(page_no)
    return None if page is None else float(page.size.height)


def to_topleft(bbox: BoundingBox, height: float) -> BBox | None:
    """Normalize to TOPLEFT `(l, t, r, b)`, or None if the box is degenerate."""
    box = bbox.to_top_left_origin(height)
    left, top, right, bottom = float(box.l), float(box.t), float(box.r), float(box.b)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def locate(doc: DoclingDocument, item: DocItem) -> tuple[int | None, BBox | None]:
    """First provenance of `item` as `(page_no, bbox)`.

    Items may carry several provenances when they straddle a page break; the
    first one is what a viewer scrolls to, which is all a citation needs.
    """
    prov = getattr(item, "prov", None)
    if not prov:
        return (None, None)
    first = prov[0]
    height = page_height(doc, first.page_no)
    if height is None:
        return (first.page_no, None)
    return (first.page_no, to_topleft(first.bbox, height))


def span_of(doc: DoclingDocument, item: DocItem) -> Span:
    """Full `Span` for an item, including its char range within the page text."""
    page_no, bbox = locate(doc, item)
    prov = getattr(item, "prov", None)
    char_start: int | None = None
    char_end: int | None = None
    if prov and prov[0].charspan is not None:
        char_start, char_end = int(prov[0].charspan[0]), int(prov[0].charspan[1])
    return Span(
        self_ref=item.self_ref,
        page_no=page_no,
        char_start=char_start,
        char_end=char_end,
        bbox=bbox,
    )
