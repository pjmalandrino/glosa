"""The port every engine flattens to.

Three engines with three trace shapes — `glosa`'s `Trace`, `docling-agent`'s
`RAGTrace`, PageIndex's tree walk — plus two controls that do not navigate at
all. `EngineAnswer` is the whole comparable surface between them, which is what
makes the report a table instead of three paragraphs.

Two shapes here are driven by what is being compared, not by taste.

**`prepare` is separate from `answer`.** PageIndex pays an LLM call per node
*before the first question*, to write the summaries its tree search navigates
on; `glosa` pays zero because it reads the host's projection. Folding that into
per-item latency would either charge all of it to item #1 or amortise it into
invisibility. Two methods, timed separately, reported as cold and warm columns.

**`Capabilities` is explicit.** A metric an engine cannot report is `None`
everywhere downstream and `—` in the report. An engine that does not say what it
read has not read nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gbench.domain.item import RenderedItem


@dataclass(frozen=True, slots=True)
class BenchDocument:
    """One corpus document, in every shape the engines need.

    All three come from the *same* conversion (`EVAL.md` §3.1): `docling_json`
    is the committed artefact, `markdown` is its `export_to_markdown()`, and
    `pdf_path` is kept for a separately-labelled end-to-end experiment that is
    deliberately not part of the headline table.
    """

    slug: str
    docling_json: str
    markdown: str
    pdf_path: str | None = None
    doc_hash: str = ""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What this engine can report about its own run."""

    read_refs: bool = False
    """Names the nodes it read — required for `hit@k` and `read_precision`."""
    abstention: bool = False
    """Has a typed signal for "not in this document"."""
    llm_calls: bool = False
    groundedness: bool = False
    """Checks its own citation against the text it read — glosa only, today."""

    abstention_signal: str = "none"
    """Human-readable mapping, printed next to the score so the reader can see
    what "abstained" meant for this engine (`EVAL.md` §3.5)."""


@dataclass(frozen=True, slots=True)
class PrepareCost:
    """What an engine spent before the first question was asked."""

    llm_calls: int | None = None
    prompt_chars: int | None = None
    wall_s: float = 0.0
    cached: bool = False
    """True when `prepare` was served from disk — its cost belongs to the cold
    column only."""


@dataclass(frozen=True, slots=True)
class EngineAnswer:
    """One question, answered. The only thing the scorer ever sees."""

    text: str
    abstained: bool = False
    read_refs: tuple[str, ...] = ()
    """In the order they were read — `hit@k` needs the order."""
    llm_calls: int | None = None
    prompt_chars: int | None = None
    wall_s: float = 0.0
    error: str = ""
    grounded: bool | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    """Opaque, engine-shaped, kept in the journal for forensics. Nothing
    downstream may branch on it — that is what `Capabilities` is for."""

    @property
    def errored(self) -> bool:
        return bool(self.error)


@runtime_checkable
class Engine(Protocol):
    """A thing that answers a rendered item over a document."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        """Everything done before the first question. Cached by document hash."""
        ...

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        """Answer one item. Must never see `item.item.answer`."""
        ...

    async def aclose(self) -> None: ...
