"""`DocIndex` — parse a `DoclingDocument` once, then answer structural questions
about it cheaply.

The whole structure is derived in a **single ordered pass** driven by real
heading levels (`TitleItem` → 0, `SectionHeaderItem.level` → 1..n) and a level
stack. That one choice handles hierarchical and flat documents identically, and
avoids the upstream bug where `iterate_items` *tree depth* is mistaken for
heading level — on a flat document depth is constant, so "stop at the next
same-or-higher heading" never fires and a section swallows the rest of the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docling_core.types.doc import (
    DocItem,
    DoclingDocument,
    SectionHeaderItem,
    TitleItem,
)

from glosa.document.excerpt import build_excerpt, build_page_excerpt, serialize_item
from glosa.document.prov import locate
from glosa.errors import DocumentParseError
from glosa.types import Excerpt, UnitKind

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_EXCERPT_BUDGET = 8_000
_TITLE_CLIP = 120


@dataclass(frozen=True, slots=True)
class Unit:
    """A retrieval unit: something the agent can decide to read.

    `ref` is always a ref that exists in the serialized document, so a consumer
    can resolve it against its own projection of the tree (Docling Studio maps
    it straight onto a Cytoscape node).
    """

    ref: str
    title: str
    level: int
    kind: UnitKind
    order: int
    page_no: int | None
    char_len: int
    child_refs: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Short human label used in the outline (`h2`, `page 3`, `document`)."""
        if self.kind is UnitKind.PAGE:
            return f"page {self.page_no}"
        if self.kind is UnitKind.DOCUMENT:
            return "document"
        return "title" if self.level == 0 else f"h{self.level}"


@dataclass
class _Building:
    """Mutable scaffold used during the single build pass."""

    ref: str
    title: str
    level: int
    order: int
    page_no: int | None
    heading: DocItem | None
    members: list[DocItem] = field(default_factory=list)
    children: list[_Building] = field(default_factory=list)


class DocIndex:
    """Structural index over one `DoclingDocument`."""

    def __init__(self, doc: DoclingDocument, *, doc_hash: str = "") -> None:
        self.doc = doc
        self.doc_hash = doc_hash
        self._rendered: dict[str, str] = {}
        self._excerpts: dict[tuple[str, int], Excerpt] = {}
        self._units: dict[str, Unit] = {}
        self._members: dict[str, list[DocItem]] = {}
        self._headings: dict[str, DocItem] = {}
        self._build()

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_json(cls, document_json: str) -> DocIndex:
        """Parse a serialized `DoclingDocument`.

        Raises `DocumentParseError` rather than letting a pydantic
        `ValidationError` escape, so callers have one exception type to map.
        """
        try:
            doc = DoclingDocument.model_validate_json(document_json)
        except Exception as exc:
            raise DocumentParseError(f"not a valid DoclingDocument: {exc}") from exc
        digest = hashlib.sha256(document_json.encode("utf-8")).hexdigest()
        return cls(doc, doc_hash=digest)

    def _render(self, item: DocItem) -> str:
        cached = self._rendered.get(item.self_ref)
        if cached is None:
            cached = serialize_item(self.doc, item)
            self._rendered[item.self_ref] = cached
        return cached

    def _build(self) -> None:
        stack: list[_Building] = []
        roots: list[_Building] = []
        flat: list[_Building] = []
        preamble: list[DocItem] = []

        for item, _ in self.doc.iterate_items():
            if not isinstance(item, DocItem):
                continue
            level = _heading_level(item)
            if level is None:
                (stack[-1].members if stack else preamble).append(item)
                continue

            while stack and stack[-1].level >= level:
                stack.pop()

            page_no, _ = locate(self.doc, item)
            node = _Building(
                ref=item.self_ref,
                title=_clip(str(getattr(item, "text", "")).strip()),
                level=level,
                order=len(flat),
                page_no=page_no,
                heading=item,
            )
            flat.append(node)
            (stack[-1].children if stack else roots).append(node)
            stack.append(node)

        if flat:
            self._build_sections(flat, preamble)
        elif self.doc.pages:
            self._build_pages()
        else:
            self._build_whole_document()

    def _build_sections(self, flat: list[_Building], preamble: list[DocItem]) -> None:
        # Content before the first heading becomes a unit anchored on its own
        # first item, so the ref stays resolvable in the caller's document tree.
        if preamble:
            head = preamble[0]
            page_no, _ = locate(self.doc, head)
            lead = _clip(self._render(head)) or "(document preamble)"
            node = _Building(
                ref=head.self_ref,
                title=lead,
                level=0,
                order=-1,
                page_no=page_no,
                heading=None,
                members=preamble,
            )
            flat.insert(0, node)

        for node in flat:
            self._members[node.ref] = node.members
            if node.heading is not None:
                self._headings[node.ref] = node.heading

        for node in flat:
            self._units[node.ref] = Unit(
                ref=node.ref,
                title=node.title,
                level=node.level,
                kind=UnitKind.SECTION,
                order=node.order,
                page_no=node.page_no,
                char_len=self._subtree_chars(node),
                child_refs=tuple(child.ref for child in node.children),
            )

    def _build_pages(self) -> None:
        for order, page_no in enumerate(sorted(self.doc.pages)):
            ref = f"#/pages/{page_no}"
            items = [
                i for i, _ in self.doc.iterate_items(page_no=page_no) if isinstance(i, DocItem)
            ]
            self._members[ref] = items
            self._units[ref] = Unit(
                ref=ref,
                title=_clip(" ".join(self._render(i) for i in items[:3]).strip())
                or f"Page {page_no}",
                level=0,
                kind=UnitKind.PAGE,
                order=order,
                page_no=page_no,
                char_len=sum(len(self._render(i)) for i in items),
            )

    def _build_whole_document(self) -> None:
        items = [i for i, _ in self.doc.iterate_items() if isinstance(i, DocItem)]
        if not items:
            return  # nothing readable: `units` stays empty and callers can say so
        self._members["#/body"] = items
        self._units["#/body"] = Unit(
            ref="#/body",
            title=self.doc.name or "Document",
            level=0,
            kind=UnitKind.DOCUMENT,
            order=0,
            page_no=None,
            char_len=sum(len(self._render(i)) for i in items),
        )

    def _subtree_chars(self, node: _Building) -> int:
        total = len(node.title) + sum(len(self._render(i)) for i in node.members)
        return total + sum(self._subtree_chars(child) for child in node.children)

    # -- queries --------------------------------------------------------------

    @property
    def units(self) -> tuple[Unit, ...]:
        """Every readable unit, in document order."""
        return tuple(sorted(self._units.values(), key=lambda u: u.order))

    @property
    def refs(self) -> frozenset[str]:
        return frozenset(self._units)

    @property
    def kind(self) -> UnitKind:
        """What the units of this document are — sections, pages, or one blob."""
        first = next(iter(self.units), None)
        return UnitKind.DOCUMENT if first is None else first.kind

    @property
    def total_chars(self) -> int:
        return sum(len(text) for text in self._rendered.values())

    @property
    def title(self) -> str:
        return self.doc.name or "Document"

    def get(self, ref: str) -> Unit | None:
        return self._units.get(ref)

    def excerpt(self, ref: str, *, char_budget: int = DEFAULT_EXCERPT_BUDGET) -> Excerpt:
        """Text of one unit, including its subsections, capped at `char_budget`."""
        key = (ref, char_budget)
        cached = self._excerpts.get(key)
        if cached is not None:
            return cached

        unit = self._units.get(ref)
        if unit is None:
            excerpt = Excerpt(ref=ref, kind=UnitKind.SECTION, text="", parts=())
        elif unit.kind is UnitKind.PAGE and unit.page_no is not None:
            excerpt = build_page_excerpt(
                self.doc,
                page_no=unit.page_no,
                char_budget=char_budget,
                render=self._render,
            )
        else:
            excerpt = build_excerpt(
                self.doc,
                ref=ref,
                kind=unit.kind,
                items=list(self._subtree_items(unit)),
                char_budget=char_budget,
                render=self._render,
            )
        self._excerpts[key] = excerpt
        return excerpt

    def full_excerpt(self, *, char_budget: int = DEFAULT_EXCERPT_BUDGET) -> Excerpt:
        """The whole document as one excerpt — the cheap path for short files.

        A two-page document does not need a navigation loop; reading it once
        costs a single call instead of the six the loop would spend.
        """
        items = [i for i, _ in self.doc.iterate_items() if isinstance(i, DocItem)]
        return build_excerpt(
            self.doc,
            ref="#/body",
            kind=UnitKind.DOCUMENT,
            items=items,
            char_budget=char_budget,
            render=self._render,
        )

    def _subtree_items(self, unit: Unit) -> Iterable[DocItem]:
        heading = self._headings.get(unit.ref)
        if heading is not None:
            yield heading
        yield from self._members.get(unit.ref, [])
        for child_ref in unit.child_refs:
            child = self._units.get(child_ref)
            if child is not None:
                yield from self._subtree_items(child)


def _heading_level(item: DocItem) -> int | None:
    """Real heading level, or None if the item is not a heading."""
    if isinstance(item, TitleItem):
        return 0
    if isinstance(item, SectionHeaderItem):
        return max(1, int(item.level))
    return None


def _clip(text: str, limit: int = _TITLE_CLIP) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
