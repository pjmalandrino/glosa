"""Core value types.

Two layers live here on purpose:

* the **native** types (`Span`, `Excerpt`, `Step`, `Trace`) carry everything
  glosa knows, including provenance we intend to surface later;
* the **legacy** types (`LegacyIteration`, `LegacyResult`) carry exactly the six
  fields Docling Studio's `ReasoningIteration` expects, so a `.model_dump()`
  splat into Studio's dataclass keeps working unchanged.

Native types are frozen dataclasses (cheap, immutable). Legacy types are
pydantic models because the receiving code calls `.model_dump()` on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel

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


@dataclass(frozen=True, slots=True)
class Span:
    """A located piece of the source document.

    `node_id` is the host's graph id (`elem::<self_ref>`, or `page::<n>`), so a
    consumer can highlight the node without re-deriving anything.
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


# --- Legacy projection -------------------------------------------------------
# Field names and order are load-bearing: Docling Studio does
# `ReasoningIteration(**it.model_dump())`. Do not rename or add fields here —
# add them to `Step` instead.


class LegacyIteration(BaseModel):
    """Wire-compatible with `docling_agent.agent.rag_models.RAGIteration`."""

    iteration: int
    section_ref: str
    reason: str
    section_text_length: int
    can_answer: bool
    response: str


class LegacyResult(BaseModel):
    """Wire-compatible with `docling_agent.agent.rag_models.RAGResult`."""

    answer: str
    iterations: list[LegacyIteration]
    converged: bool
