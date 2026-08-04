"""The document object glosa reads: Docling Studio's projection of it.

`StudioProjection` is the same node set, the same ids, the same reading order
and the same section scoping that Studio's graph view and canvas overlay use:

* nodes are `iter_items` minus `skip_refs`, so InlineGroup style runs and
  picture-internal labels never appear;
* the id of a node is `elem::<self_ref>`, matching `infra/docling_graph.py`;
* reading order is `dfs_order` — the NEXT chain the graph draws;
* a **section** runs from one `SectionHeader` to the next along that chain,
  which is exactly the rule the frontend's `computeSectionParents` applies to
  build its compound nodes. Note there is deliberately *no* level nesting: an
  `h2` after an `h1` starts a new scope rather than nesting inside it, because
  that is what the UI shows. Levels are kept for display in the outline only.

Consequence: every ref glosa can put in a trace resolves to a node the UI
already has, and the set of nodes a step claims to have read is the set the UI
highlights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from glosa.errors import DocumentParseError
from glosa.studio.render import render_item
from glosa.studio.tree import (
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
from glosa.types import BBox

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from glosa.studio.ports import TreeReader

NODE_PREFIX = "elem::"
PAGE_PREFIX = "page::"


def node_id_for(self_ref: str) -> str:
    """Cytoscape id of an element node, as `infra/docling_graph.py` builds it."""
    return f"{NODE_PREFIX}{self_ref}"


def page_node_id(page_no: int) -> str:
    return f"{PAGE_PREFIX}{page_no}"


@dataclass(frozen=True, slots=True)
class Element:
    """One node of the projected document."""

    self_ref: str
    node_id: str
    docling_label: str
    graph_label: str
    text: str
    order: int
    level: int | None = None
    page_no: int | None = None
    bbox: BBox | None = None
    provs: tuple[dict[str, Any], ...] = ()
    parent: str | None = None
    is_section: bool = False
    is_furniture: bool = False

    @property
    def pages(self) -> tuple[int, ...]:
        seen: dict[int, None] = {}
        for prov in self.provs:
            page = prov.get("page_no")
            if isinstance(page, int):
                seen[page] = None
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class Scope:
    """A section: its heading (if any) and the elements that belong to it."""

    anchor: Element
    members: tuple[Element, ...]

    @property
    def ref(self) -> str:
        return self.anchor.self_ref

    @property
    def elements(self) -> tuple[Element, ...]:
        """Anchor first, then members.

        For a section the anchor is the heading; for content that precedes the
        first heading it is that content's own first element. Either way the
        anchor is part of what gets read.
        """
        return (self.anchor, *self.members)


class StudioProjection:
    """Parsed, collapsed, ordered view of a `document_json` blob."""

    def __init__(self, doc_data: dict[str, Any], *, tree_reader: TreeReader | None = None) -> None:
        self.doc_data = doc_data
        self._reader: TreeReader = tree_reader or BundledTreeReader()
        self.pages: dict[int, float | None] = {}
        self.page_widths: dict[int, float | None] = {}
        self.elements: tuple[Element, ...] = ()
        self.by_ref: dict[str, Element] = {}
        self._build()

    @classmethod
    def from_json(
        cls, document_json: str, *, tree_reader: TreeReader | None = None
    ) -> StudioProjection:
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
            self.pages[page_no] = _as_float(page.get("height"))
            self.page_widths[page_no] = _as_float(page.get("width"))

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
            element = Element(
                self_ref=ref,
                node_id=node_id_for(ref),
                docling_label=label,
                graph_label=element_label(label),
                text=text,
                order=order,
                level=level,
                page_no=page_no,
                bbox=bbox,
                provs=tuple(provs),
                parent=parent_ref(item),
                is_section=is_section_header(item),
                is_furniture=_is_furniture(item, label),
            )
            elements.append(element)
            self.by_ref[ref] = element

        self.elements = tuple(elements)

    def _locate(self, provs: Sequence[dict[str, Any]]) -> tuple[int | None, BBox | None]:
        if not provs:
            return (None, None)
        first = provs[0]
        page_no = first.get("page_no")
        page_no = page_no if isinstance(page_no, int) else None
        height = self.pages.get(page_no) if page_no is not None else None
        return page_no, to_topleft(first, height)

    # -- queries --------------------------------------------------------------

    @property
    def title(self) -> str:
        name = self.doc_data.get("name")
        return str(name) if name else "Document"

    @property
    def node_ids(self) -> frozenset[str]:
        """Every `elem::` id in the projection — what a trace may reference."""
        return frozenset(element.node_id for element in self.elements)

    @property
    def has_sections(self) -> bool:
        return any(element.is_section for element in self.elements)

    def readable(self, *, include_furniture: bool = False) -> tuple[Element, ...]:
        """Elements worth putting in front of a model.

        Running heads and footers stay in the projection — they are real nodes
        and must remain addressable — but repeating them in every excerpt is
        noise, so they are out by default.
        """
        if include_furniture:
            return self.elements
        return tuple(e for e in self.elements if not e.is_furniture)

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

    def iter_page_numbers(self) -> Iterator[int]:
        yield from sorted(self.pages)


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
