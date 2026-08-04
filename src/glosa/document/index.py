"""`DocIndex` — retrieval units over Docling Studio's document projection.

Every unit is anchored on a node that exists in the host's graph, and every
excerpt records the exact node ids it was built from. Nothing here re-derives
document structure: that all comes from `glosa.studio.projection`, which
mirrors Studio's own collapse rules, reading order and section scoping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.studio.projection import StudioProjection, page_node_id
from glosa.types import Excerpt, ExcerptPart, UnitKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.studio.ports import TreeReader
    from glosa.studio.projection import Element

DEFAULT_EXCERPT_BUDGET = 8_000
TRUNCATION_MARKER = "\n\n[… excerpt truncated to fit the context budget …]"
_TITLE_CLIP = 120
_MIN_PARTIAL_CHARS = 200


@dataclass(frozen=True, slots=True)
class Unit:
    """Something the agent can decide to read.

    `ref` and `node_id` both address the host's graph — `ref` is the Docling
    `self_ref` the legacy trace field carries, `node_id` is the Cytoscape id.
    """

    ref: str
    node_id: str
    title: str
    kind: UnitKind
    order: int
    char_len: int
    graph_label: str = ""
    level: int | None = None
    page_no: int | None = None
    pages: tuple[int, ...] = ()
    element_refs: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Short human label for the outline."""
        if self.kind is UnitKind.PAGE:
            return f"page {self.page_no}"
        if self.kind is UnitKind.DOCUMENT:
            return "document"
        if self.level is None:
            return "section"
        return "title" if self.level == 0 else f"h{self.level}"


class DocIndex:
    """Units, excerpts and caching over one projected document."""

    def __init__(
        self,
        projection: StudioProjection,
        *,
        doc_hash: str = "",
        include_furniture: bool = False,
    ) -> None:
        self.projection = projection
        self.doc_hash = doc_hash
        self._include_furniture = include_furniture
        self._units: dict[str, Unit] = {}
        self._elements: dict[str, tuple[Element, ...]] = {}
        self._excerpts: dict[tuple[str, int], Excerpt] = {}
        self._build()

    @classmethod
    def from_json(
        cls,
        document_json: str,
        *,
        tree_reader: TreeReader | None = None,
        include_furniture: bool = False,
    ) -> DocIndex:
        """Parse a `document_json` blob into an index.

        Raises `DocumentParseError` on anything that is not a serialized
        `DoclingDocument`, so callers have one exception type to map.
        """
        projection = StudioProjection.from_json(document_json, tree_reader=tree_reader)
        digest = hashlib.sha256(document_json.encode("utf-8")).hexdigest()
        return cls(projection, doc_hash=digest, include_furniture=include_furniture)

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        readable = self.projection.readable(include_furniture=self._include_furniture)
        if not readable:
            return
        if self.projection.has_sections:
            self._build_sections()
            return
        if self.projection.pages:
            self._build_pages()
            if self._units:
                return
        self._build_whole_document(readable)

    def _build_sections(self) -> None:
        scopes = self.projection.scopes(include_furniture=self._include_furniture)
        for order, scope in enumerate(scopes):
            elements = scope.elements
            anchor = scope.anchor
            self._elements[scope.ref] = elements
            self._units[scope.ref] = Unit(
                ref=scope.ref,
                node_id=anchor.node_id,
                title=_clip(anchor.text) or "(untitled)",
                kind=UnitKind.SECTION,
                order=order,
                char_len=sum(len(e.text) for e in elements),
                graph_label=anchor.graph_label,
                level=anchor.level if anchor.is_section else None,
                page_no=anchor.page_no,
                pages=_pages_of(elements),
                element_refs=tuple(e.self_ref for e in elements),
            )

    def _build_pages(self) -> None:
        for order, page_no in enumerate(self.projection.iter_page_numbers()):
            elements = self.projection.page_elements(
                page_no, include_furniture=self._include_furniture
            )
            if not elements:
                continue
            ref = f"#/pages/{page_no}"
            self._elements[ref] = elements
            self._units[ref] = Unit(
                ref=ref,
                node_id=page_node_id(page_no),
                title=_clip(" ".join(e.text for e in elements[:3])) or f"Page {page_no}",
                kind=UnitKind.PAGE,
                order=order,
                char_len=sum(len(e.text) for e in elements),
                graph_label="Page",
                page_no=page_no,
                pages=(page_no,),
                element_refs=tuple(e.self_ref for e in elements),
            )

    def _build_whole_document(self, elements: Sequence[Element]) -> None:
        anchor = elements[0]
        self._elements[anchor.self_ref] = tuple(elements)
        self._units[anchor.self_ref] = Unit(
            ref=anchor.self_ref,
            node_id=anchor.node_id,
            title=self.projection.title,
            kind=UnitKind.DOCUMENT,
            order=0,
            char_len=sum(len(e.text) for e in elements),
            graph_label=anchor.graph_label,
            pages=_pages_of(elements),
            element_refs=tuple(e.self_ref for e in elements),
        )

    # -- queries --------------------------------------------------------------

    @property
    def units(self) -> tuple[Unit, ...]:
        return tuple(sorted(self._units.values(), key=lambda u: u.order))

    @property
    def refs(self) -> frozenset[str]:
        return frozenset(self._units)

    @property
    def kind(self) -> UnitKind:
        first = next(iter(self.units), None)
        return UnitKind.DOCUMENT if first is None else first.kind

    @property
    def total_chars(self) -> int:
        readable = self.projection.readable(include_furniture=self._include_furniture)
        return sum(len(e.text) for e in readable)

    @property
    def title(self) -> str:
        return self.projection.title

    def get(self, ref: str) -> Unit | None:
        return self._units.get(ref)

    def excerpt(self, ref: str, *, char_budget: int = DEFAULT_EXCERPT_BUDGET) -> Excerpt:
        """Text of one unit, capped at `char_budget`."""
        key = (ref, char_budget)
        cached = self._excerpts.get(key)
        if cached is not None:
            return cached

        unit = self._units.get(ref)
        if unit is None:
            excerpt = Excerpt(ref=ref, kind=UnitKind.SECTION, text="", parts=())
        else:
            excerpt = _assemble(
                ref=ref,
                kind=unit.kind,
                elements=self._elements.get(ref, ()),
                char_budget=char_budget,
            )
        self._excerpts[key] = excerpt
        return excerpt

    def full_excerpt(self, *, char_budget: int = DEFAULT_EXCERPT_BUDGET) -> Excerpt:
        """The whole document as one excerpt — the cheap path for short files.

        Anchored on the first projected element so the ref still resolves in
        the host's graph; the node ids of everything read travel with it.
        """
        elements = self.projection.readable(include_furniture=self._include_furniture)
        ref = elements[0].self_ref if elements else "#/body"
        return _assemble(
            ref=ref, kind=UnitKind.DOCUMENT, elements=elements, char_budget=char_budget
        )


def _assemble(
    *, ref: str, kind: UnitKind, elements: Sequence[Element], char_budget: int
) -> Excerpt:
    parts: list[ExcerptPart] = []
    chunks: list[str] = []
    used = 0
    truncated = False

    for element in elements:
        text = element.text
        if not text:
            continue
        if used + len(text) > char_budget:
            remaining = char_budget - used
            # Keep a partial head only when it is long enough to carry meaning.
            if remaining > _MIN_PARTIAL_CHARS:
                text = text[:remaining]
                parts.append(_part(element, text))
                chunks.append(text)
            truncated = True
            break
        parts.append(_part(element, text))
        chunks.append(text)
        used += len(text)

    body = "\n\n".join(chunks)
    if truncated:
        body += TRUNCATION_MARKER
    return Excerpt(ref=ref, kind=kind, text=body, parts=tuple(parts), truncated=truncated)


def _part(element: Element, text: str) -> ExcerptPart:
    return ExcerptPart(
        self_ref=element.self_ref,
        node_id=element.node_id,
        text=text,
        page_no=element.page_no,
        bbox=element.bbox,
        graph_label=element.graph_label,
    )


def _pages_of(elements: Sequence[Element]) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for element in elements:
        for page in element.pages:
            seen[page] = None
    return tuple(sorted(seen))


def _clip(text: str, limit: int = _TITLE_CLIP) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
