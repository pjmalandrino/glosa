"""Docling adapter: a serialized `DoclingDocument` → a `DocumentProjection`.

This is where every Docling- and Studio-specific assumption lives, and the only
place allowed to know them:

* nodes are `iter_items` minus `skip_refs`, so InlineGroup style runs and
  picture-internal labels never appear;
* the id of a node is `elem::<self_ref>`, matching `infra/docling_graph.py`;
* reading order is `dfs_order` — the NEXT chain the graph draws;
* a **section** runs from one `SectionHeader` to the next along that chain,
  which is exactly the rule the frontend's `computeSectionParents` applies to
  build its compound nodes. There is deliberately *no* level nesting: an `h2`
  after an `h1` starts a new scope rather than nesting inside it, because that
  is what the UI shows. Levels survive for display in the outline only.

Consequence: every ref the domain can put in a trace resolves to a node the UI
already has, and the set of nodes a step claims to have read is the set the UI
highlights.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from glosa.domain.errors import DocumentParseError
from glosa.domain.values import Element, Scope
from glosa.infra.docling.render import render_item
from glosa.infra.docling.tree import (
    FURNITURE_LABELS,
    BundledTreeReader,
    dfs_order,
    element_label,
    is_section_header,
    item_label,
    iter_pages,
    iter_provs,
    parent_ref,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.values import BBox
    from glosa.ports.document import DocumentProjection, TreeReader

NODE_PREFIX = "elem::"
PAGE_PREFIX = "page::"
PAGE_REF_PREFIX = "#/pages/"


def node_id_for(self_ref: str) -> str:
    """Cytoscape id of an element node, as `infra/docling_graph.py` builds it."""
    return f"{NODE_PREFIX}{self_ref}"


def page_node_id(page_no: int) -> str:
    return f"{PAGE_PREFIX}{page_no}"


def page_ref(page_no: int) -> str:
    return f"{PAGE_REF_PREFIX}{page_no}"


class DoclingProjection:
    """Parsed, collapsed, ordered view of a `document_json` blob.

    Implements `glosa.ports.document.DocumentProjection`.
    """

    def __init__(self, doc_data: dict[str, Any], *, tree_reader: TreeReader | None = None) -> None:
        self.doc_data = doc_data
        self._reader: TreeReader = tree_reader or BundledTreeReader()
        self._page_heights: dict[int, float | None] = {}
        self._page_widths: dict[int, float | None] = {}
        self._elements: tuple[Element, ...] = ()
        self.by_ref: dict[str, Element] = {}
        self._build()

    @classmethod
    def from_json(
        cls, document_json: str, *, tree_reader: TreeReader | None = None
    ) -> DoclingProjection:
        try:
            doc_data = json.loads(document_json)
        except (TypeError, ValueError) as exc:
            raise DocumentParseError(f"document_json is not valid JSON: {exc}") from exc
        if not isinstance(doc_data, dict):
            raise DocumentParseError("document_json must decode to an object")
        return cls(doc_data, tree_reader=tree_reader)

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        for page in iter_pages(self.doc_data):
            page_no = page["page_no"]
            self._page_heights[page_no] = _as_float(page.get("height"))
            self._page_widths[page_no] = _as_float(page.get("width"))

        skip_refs, inline_meta = self._reader.build_collapse_index(self.doc_data)
        raw_by_ref: dict[str, dict[str, Any]] = {}
        for _, item in self._reader.iter_items(self.doc_data):
            ref = item.get("self_ref")
            if ref and ref not in skip_refs:
                raw_by_ref[ref] = item

        # Reading order first, then anything unreachable from `body` so no
        # projected node is silently unaddressable.
        ordered = [ref for ref in dfs_order(self.doc_data, skip_refs) if ref in raw_by_ref]
        seen = set(ordered)
        ordered += [ref for ref in raw_by_ref if ref not in seen]

        elements: list[Element] = []
        for order, ref in enumerate(ordered):
            item = raw_by_ref[ref]
            meta = inline_meta.get(ref)
            provs = list(meta["provs"]) if meta is not None else iter_provs(item)
            text = render_item(
                item,
                by_ref=raw_by_ref,
                inline_text=meta["text"] if meta is not None else None,
            )
            label = item_label(item)
            page_no, bbox = self._locate(provs)
            # `TitleItem` carries no `level`; it is the document's top heading.
            level = _as_int(item.get("level"))
            if level is None and label == "title":
                level = 0
            summary, keywords = _enrichment(item)
            element = Element(
                self_ref=ref,
                node_id=node_id_for(ref),
                text=text,
                order=order,
                docling_label=label,
                graph_label=element_label(label),
                level=level,
                page_no=page_no,
                pages=_pages_of(provs),
                bbox=bbox,
                provs=tuple(provs),
                summary=summary,
                keywords=keywords,
                parent=parent_ref(item),
                is_section=is_section_header(item),
                is_furniture=_is_furniture(item, label),
            )
            elements.append(element)
            self.by_ref[ref] = element

        self._elements = tuple(elements)

    def _locate(self, provs: Sequence[dict[str, Any]]) -> tuple[int | None, BBox | None]:
        if not provs:
            return (None, None)
        first = provs[0]
        page_no = first.get("page_no")
        page_no = page_no if isinstance(page_no, int) else None
        height = self._page_heights.get(page_no) if page_no is not None else None
        return page_no, to_topleft(first, height)

    # -- DocumentProjection ---------------------------------------------------

    @property
    def title(self) -> str:
        name = self.doc_data.get("name")
        return str(name) if name else "Document"

    @property
    def elements(self) -> tuple[Element, ...]:
        return self._elements

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(element.node_id for element in self._elements)

    @property
    def has_sections(self) -> bool:
        return any(element.is_section for element in self._elements)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self._page_heights))

    def readable(self, *, include_furniture: bool = False) -> tuple[Element, ...]:
        """Elements worth putting in front of a model.

        Running heads and footers stay in the projection — they are real nodes
        and must remain addressable — but repeating them in every excerpt is
        noise, so they are out by default.
        """
        if include_furniture:
            return self._elements
        return tuple(e for e in self._elements if not e.is_furniture)

    def scopes(self, *, include_furniture: bool = False) -> tuple[Scope, ...]:
        """Section scopes, following the NEXT chain exactly as the UI does."""
        scopes: list[Scope] = []
        anchor: Element | None = None
        members: list[Element] = []

        for element in self.readable(include_furniture=include_furniture):
            if element.is_section:
                if anchor is not None:
                    scopes.append(Scope(anchor, tuple(members)))
                elif members:
                    # Content before the first heading: anchor it on its own
                    # first element so the ref still resolves in the graph.
                    scopes.append(Scope(members[0], tuple(members[1:])))
                anchor, members = element, []
                continue
            members.append(element)

        if anchor is not None:
            scopes.append(Scope(anchor, tuple(members)))
        elif members:
            scopes.append(Scope(members[0], tuple(members[1:])))
        return tuple(scopes)

    def page_elements(
        self, page_no: int, *, include_furniture: bool = False
    ) -> tuple[Element, ...]:
        return tuple(
            e for e in self.readable(include_furniture=include_furniture) if page_no in e.pages
        )

    def page_ref(self, page_no: int) -> str:
        return page_ref(page_no)

    def page_node_id(self, page_no: int) -> str:
        return page_node_id(page_no)


class DoclingProjector:
    """`DocumentProjector` over serialized `DoclingDocument` payloads.

    Args:
        tree_reader: The host's own tree reader. Docling Studio wires one at
            `main.py:321`; passing it means the collapse rules have a single
            implementation in the deployment.
    """

    def __init__(self, *, tree_reader: TreeReader | None = None) -> None:
        self._tree_reader = tree_reader

    def project(self, document_json: str) -> DocumentProjection:
        return DoclingProjection.from_json(document_json, tree_reader=self._tree_reader)


def to_topleft(prov: dict[str, Any], page_height: float | None) -> BBox | None:
    """Normalize one prov row to TOPLEFT `(l, t, r, b)`.

    Mirrors Studio's `infra/bbox.py`: BOTTOMLEFT is flipped against the page
    height, and a degenerate rectangle is dropped. Studio substitutes its
    `EMPTY_BBOX` sentinel at the wire edge; glosa returns None so "unknown
    location" stays distinguishable from "located at nothing".
    """
    left = _as_float(prov.get("bbox_l")) or 0.0
    top = _as_float(prov.get("bbox_t")) or 0.0
    right = _as_float(prov.get("bbox_r")) or 0.0
    bottom = _as_float(prov.get("bbox_b")) or 0.0
    origin = str(prov.get("coord_origin") or "TOPLEFT").upper()

    if origin == "BOTTOMLEFT":
        if page_height is None:
            return None
        top, bottom = page_height - top, page_height - bottom

    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _enrichment(item: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Pull `meta.summary.text` and `meta.keywords.values` when the host's
    enrichment pipeline produced them. Absent enrichment, both are empty and
    the lexical prior carries the whole load."""
    meta = item.get("meta")
    if not isinstance(meta, dict):
        return ("", ())

    summary = ""
    node = meta.get("summary")
    if isinstance(node, dict) and isinstance(node.get("text"), str):
        summary = node["text"].strip()

    keywords: tuple[str, ...] = ()
    node = meta.get("keywords")
    if isinstance(node, dict) and isinstance(node.get("values"), list):
        keywords = tuple(str(v).strip() for v in node["values"] if str(v).strip())

    return summary, keywords


def _pages_of(provs: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for prov in provs:
        page = prov.get("page_no")
        if isinstance(page, int):
            seen[page] = None
    return tuple(seen)


def _is_furniture(item: dict[str, Any], label: str) -> bool:
    if str(item.get("content_layer") or "").lower() == "furniture":
        return True
    return label in FURNITURE_LABELS


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
