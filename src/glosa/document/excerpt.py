"""Turning document items into the text a model actually reads.

Two things differ from the upstream chunkless loop:

* tables are serialized as HTML everywhere, not only in page mode — flattening
  a table to markdown loses row/column association, which is exactly what
  numeric questions need;
* truncation is explicit. An excerpt that hit the budget says so, both in the
  returned object and in the text handed to the model, so a partial read is
  never silently presented as a complete one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import DocItem, PictureItem, TableItem

from glosa.document.prov import locate
from glosa.types import Excerpt, ExcerptPart, UnitKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from docling_core.types.doc import DoclingDocument

    Renderer = Callable[[DocItem], str]

TRUNCATION_MARKER = "\n\n[… excerpt truncated to fit the context budget …]"


def serialize_item(doc: DoclingDocument, item: DocItem) -> str:
    """Render a single item as text suitable for a prompt."""
    if isinstance(item, TableItem):
        try:
            return item.export_to_html(doc=doc, add_caption=True).strip()
        except Exception:
            return item.caption_text(doc).strip()
    if isinstance(item, PictureItem):
        parts = [item.caption_text(doc).strip()]
        for annotation in item.annotations:
            text = getattr(annotation, "text", None)
            if text:
                parts.append(str(text).strip())
        rendered = "\n".join(p for p in parts if p)
        return f"[figure] {rendered}" if rendered else ""
    text = getattr(item, "text", "")
    return str(text).strip()


def build_excerpt(
    doc: DoclingDocument,
    *,
    ref: str,
    kind: UnitKind,
    items: list[DocItem],
    char_budget: int,
    render: Renderer | None = None,
) -> Excerpt:
    """Assemble an `Excerpt` from an ordered list of items, honouring a budget.

    `render` lets the caller supply a memoized serializer — `DocIndex` does, so
    a table's HTML export happens once per document rather than once per read.
    """
    renderer = render or (lambda item: serialize_item(doc, item))
    parts: list[ExcerptPart] = []
    chunks: list[str] = []
    used = 0
    truncated = False

    for item in items:
        text = renderer(item)
        if not text:
            continue
        if used + len(text) > char_budget:
            remaining = char_budget - used
            # Keep a partial head only when it is long enough to carry meaning.
            if remaining > 200:
                text = text[:remaining]
                page_no, bbox = locate(doc, item)
                parts.append(ExcerptPart(item.self_ref, text, page_no, bbox))
                chunks.append(text)
            truncated = True
            break
        page_no, bbox = locate(doc, item)
        parts.append(ExcerptPart(item.self_ref, text, page_no, bbox))
        chunks.append(text)
        used += len(text)

    body = "\n\n".join(chunks)
    if truncated:
        body += TRUNCATION_MARKER
    return Excerpt(ref=ref, kind=kind, text=body, parts=tuple(parts), truncated=truncated)


def build_page_excerpt(
    doc: DoclingDocument,
    *,
    page_no: int,
    char_budget: int,
    render: Renderer | None = None,
) -> Excerpt:
    """Serialize a whole page, tables included, as one excerpt.

    Used when the document has no headings at all — the case where the upstream
    loop gives up and returns the entire document as the answer.
    """
    params = MarkdownParams(pages={page_no}, compact_tables=True, image_placeholder="")
    serializer = MarkdownDocSerializer(doc=doc, params=params)
    text = serializer.serialize().text.strip()
    truncated = len(text) > char_budget
    if truncated:
        text = text[:char_budget] + TRUNCATION_MARKER

    renderer = render or (lambda item: serialize_item(doc, item))
    parts = tuple(
        ExcerptPart(item.self_ref, renderer(item), *locate(doc, item))
        for item, _ in doc.iterate_items(page_no=page_no)
        if isinstance(item, DocItem)
    )
    return Excerpt(
        ref=f"#/pages/{page_no}",
        kind=UnitKind.PAGE,
        text=text,
        parts=tuple(p for p in parts if p.text),
        truncated=truncated,
    )
