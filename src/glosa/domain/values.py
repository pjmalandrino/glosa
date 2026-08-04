"""Domain value types.

Pure data: no I/O, no framework, and — importantly — no knowledge of how the
host serializes a document. `Element` carries a `node_id` and a `bbox` because
the domain needs to *say where something is*; it never learns how those were
derived from Docling's JSON. That stays in `glosa.infra.docling`.

`provs` is the one deliberate exception: opaque, host-shaped provenance rows
passed straight through for consumers that want them. Nothing in the domain
looks inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

BBox = tuple[float, float, float, float]
"""``(left, top, right, bottom)`` in TOPLEFT origin — the convention Docling
Studio's canvas overlay expects (cf. its ``infra/bbox.py``)."""


class RunStatus(StrEnum):
    """How a run ended.

    Replaces the single `converged` boolean, which conflated four very
    different outcomes. `converged` is still derived from this for the legacy
    projection.
    """

    ANSWERED = "answered"
    NOT_IN_DOCUMENT = "not_in_document"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def converged(self) -> bool:
        """Legacy convergence flag: only a real answer counts."""
        return self is RunStatus.ANSWERED


class UnitKind(StrEnum):
    """What kind of retrieval unit a ref points at."""

    SECTION = "section"
    PAGE = "page"
    DOCUMENT = "document"


# --- the document ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """One node of the projected document.

    Built by a `DocumentProjector`; the domain only reads it.
    """

    self_ref: str
    node_id: str
    text: str
    order: int
    docling_label: str = ""
    graph_label: str = ""
    level: int | None = None
    page_no: int | None = None
    pages: tuple[int, ...] = ()
    bbox: BBox | None = None
    provs: tuple[Mapping[str, Any], ...] = ()
    """Opaque host provenance rows, passed through untouched."""
    summary: str = ""
    """Enrichment written by the host's pipeline, when it ran.

    This is the signal PageIndex builds its whole index around and that
    `docling-agent`'s enricher writes into `meta.summary`. It costs an LLM call
    per node to produce, so glosa never generates it — but when the host has,
    ignoring it would be throwing away the best retrieval signal available."""
    keywords: tuple[str, ...] = ()
    parent: str | None = None
    is_section: bool = False
    is_furniture: bool = False


@dataclass(frozen=True, slots=True)
class Scope:
    """A section: its anchor and the elements that belong to it."""

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


# --- what a run produces -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """A located piece of the source document.

    `node_id` is the host's graph id, so a consumer can highlight the node
    without re-deriving anything.
    """

    self_ref: str
    node_id: str = ""
    page_no: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: BBox | None = None


@dataclass(frozen=True, slots=True)
class ExcerptPart:
    """One projected element inside an excerpt, kept addressable.

    Excerpts are assembled from parts rather than a flat string so that a later
    phase can align an answer sentence back to a node + bbox without re-walking
    the document.
    """

    self_ref: str
    node_id: str
    text: str
    page_no: int | None = None
    bbox: BBox | None = None
    graph_label: str = ""


@dataclass(frozen=True, slots=True)
class Excerpt:
    """The text handed to the model for one retrieval unit."""

    ref: str
    kind: UnitKind
    text: str
    parts: tuple[ExcerptPart, ...] = ()
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.text)

    @property
    def pages(self) -> tuple[int, ...]:
        seen: dict[int, None] = {}
        for part in self.parts:
            if part.page_no is not None:
                seen[part.page_no] = None
        return tuple(seen)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Host graph ids of everything in this excerpt."""
        return tuple(part.node_id for part in self.parts if part.node_id)


@dataclass(frozen=True, slots=True)
class Step:
    """One read of the document: what was chosen, why, and what came out."""

    index: int
    ref: str
    reason: str
    excerpt_chars: int
    sufficient: bool
    response: str
    kind: UnitKind = UnitKind.SECTION
    title: str = ""
    pages: tuple[int, ...] = ()
    spans: tuple[Span, ...] = ()
    node_ids: tuple[str, ...] = ()
    """Host graph ids actually read at this step — what the UI should light up.

    The legacy `section_ref` names only the anchor; this names the whole set,
    so no consumer has to re-derive section membership and risk disagreeing
    with what was really read."""
    revisited: bool = False
    fallback: bool = False
    """True when the model failed to pick a valid ref and glosa chose for it."""


@dataclass(frozen=True, slots=True)
class Trace:
    """Everything a run produced."""

    query: str
    answer: str
    status: RunStatus
    model_id: str
    steps: tuple[Step, ...] = ()
    llm_calls: int = 0
    elapsed_s: float = 0.0
    notes: tuple[str, ...] = field(default=())

    @property
    def converged(self) -> bool:
        return self.status.converged
