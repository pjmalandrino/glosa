"""`DocIndex` — retrieval units over a projected document.

Pure domain logic: how to slice a document into things worth reading, how big
each one is, how to cut an excerpt to a budget. It works against the
`DocumentProjection` port and never learns what produced it — no JSON, no
Docling, no host-specific id format (page refs and node ids come from the
projection, which owns those conventions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.lexical import Bm25Index
from glosa.domain.values import Excerpt, ExcerptPart, UnitKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.rank import UnitRanker
    from glosa.domain.values import Element
    from glosa.ports.document import DocumentProjection

DEFAULT_EXCERPT_BUDGET = 8_000
TRUNCATION_MARKER = "\n\n[… excerpt truncated to fit the context budget …]"
SELECTION_MARKER = (
    "\n\n[… {n} passage(s) of this section omitted; the ones matching the question are shown …]"
)
_TITLE_CLIP = 120
_MIN_PARTIAL_CHARS = 200


@dataclass(frozen=True, slots=True)
class Unit:
    """Something the agent can decide to read.

    `ref` and `node_id` both address the host's graph — `ref` is the ref the
    legacy trace field carries, `node_id` is the graph node id.
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
        projection: DocumentProjection,
        *,
        include_furniture: bool = False,
    ) -> None:
        self.projection = projection
        self._include_furniture = include_furniture
        self._units: dict[str, Unit] = {}
        self._elements: dict[str, tuple[Element, ...]] = {}
        self._excerpts: dict[tuple[str, int, str], Excerpt] = {}
        self._ranker: UnitRanker | None = None
        self._build()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        readable = self.projection.readable(include_furniture=self._include_furniture)
        if not readable:
            return
        if self.projection.has_sections:
            self._build_sections()
            return
        if self.projection.page_numbers:
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
        for order, page_no in enumerate(self.projection.page_numbers):
            elements = self.projection.page_elements(
                page_no, include_furniture=self._include_furniture
            )
            if not elements:
                continue
            ref = self.projection.page_ref(page_no)
            self._elements[ref] = elements
            self._units[ref] = Unit(
                ref=ref,
                node_id=self.projection.page_node_id(page_no),
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

    @property
    def ranker(self) -> UnitRanker:
        """Lexical shortlist over these units, built once per document.

        The indexes depend on the document, not on the question, so a host that
        asks several questions of the same analysis pays for them once.
        """
        if self._ranker is None:
            from glosa.domain.rank import UnitRanker

            self._ranker = UnitRanker(self)
        return self._ranker

    def get(self, ref: str) -> Unit | None:
        return self._units.get(ref)

    def excerpt(
        self,
        ref: str,
        *,
        char_budget: int = DEFAULT_EXCERPT_BUDGET,
        focus: str | None = None,
    ) -> Excerpt:
        """Text of one unit, capped at `char_budget`.

        `focus` is the question being answered. It only matters when the unit
        does not fit: instead of keeping the first N characters and hoping the
        answer is near the top, the passages that lexically match the question
        are kept, in reading order. A 40-page appendix stops being a coin flip.
        """
        key = (ref, char_budget, focus or "")
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
                focus=focus,
            )
        self._excerpts[key] = excerpt
        return excerpt

    def full_excerpt(self, *, char_budget: int = DEFAULT_EXCERPT_BUDGET) -> Excerpt:
        """The whole document as one excerpt — the cheap path for short files.

        Anchored on the first projected element so the ref still resolves in
        the host's graph; the node ids of everything read travel with it.
        """
        elements = self.projection.readable(include_furniture=self._include_furniture)
        ref = elements[0].self_ref if elements else ""
        return _assemble(
            ref=ref, kind=UnitKind.DOCUMENT, elements=elements, char_budget=char_budget
        )


def _assemble(
    *,
    ref: str,
    kind: UnitKind,
    elements: Sequence[Element],
    char_budget: int,
    focus: str | None = None,
) -> Excerpt:
    if focus and sum(len(e.text) for e in elements) > char_budget:
        return _assemble_by_relevance(
            ref=ref, kind=kind, elements=elements, char_budget=char_budget, focus=focus
        )

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


def _assemble_by_relevance(
    *, ref: str, kind: UnitKind, elements: Sequence[Element], char_budget: int, focus: str
) -> Excerpt:
    """Pack the passages that match `focus`, then restore reading order."""
    index = Bm25Index((e.self_ref, e.text) for e in elements if e.text)
    ranked = {hit.ref: hit.score for hit in index.rank(focus)}
    if not ranked:
        # No lexical signal inside the section: head-first is as good a guess
        # as any, and pretending otherwise would be theatre.
        return _assemble(ref=ref, kind=kind, elements=elements, char_budget=char_budget)

    order = {e.self_ref: i for i, e in enumerate(elements)}
    by_relevance = sorted(
        (e for e in elements if e.text),
        key=lambda e: (-ranked.get(e.self_ref, 0.0), order[e.self_ref]),
    )

    kept: list[Element] = []
    used = 0
    for element in by_relevance:
        if used + len(element.text) > char_budget:
            continue
        kept.append(element)
        used += len(element.text)

    if not kept:  # a single element larger than the whole budget
        return _assemble(ref=ref, kind=kind, elements=elements, char_budget=char_budget)

    kept.sort(key=lambda e: order[e.self_ref])
    omitted = sum(1 for e in elements if e.text) - len(kept)
    parts = tuple(_part(e, e.text) for e in kept)
    body = "\n\n".join(e.text for e in kept)
    if omitted:
        body += SELECTION_MARKER.format(n=omitted)
    return Excerpt(ref=ref, kind=kind, text=body, parts=parts, truncated=bool(omitted))


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
