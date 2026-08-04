"""`ReasoningRunner` implementation for Docling Studio — the driving adapter.

Satisfies Studio's `domain.ports.ReasoningRunner` structurally: glosa never
imports Studio, and Studio never imports glosa's types. The three `*_factory`
arguments let the host hand over its own domain classes, so the runner returns
`ReasoningResult` / `ReasoningIteration` / `ReasoningParseError` directly and no
translation adapter is needed on either side.

Being the outermost layer, this module is allowed to know about `infra` — it
supplies the default `DoclingProjector`. Everything it hands to the domain is a
port.

Wiring, in Studio's `main.py`:

    from glosa import GlosaReasoningRunner, OllamaChatModel, DoclingProjector
    from domain.ports import ReasoningParseError
    from domain.value_objects import ReasoningIteration, ReasoningResult

    app.state.reasoning_runner = GlosaReasoningRunner(
        model=OllamaChatModel(base_url=settings.ollama_host,
                              model_id=settings.reasoning_model_id),
        # the DoclingTreeReader already wired at main.py:321
        projector=DoclingProjector(tree_reader=app.state.tree_reader),
        result_factory=ReasoningResult,
        iteration_factory=ReasoningIteration,
        parse_error_factory=ReasoningParseError,
    )
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol

from glosa.adapters.legacy import (
    IterationFactory,
    LegacyIteration,
    LegacyResult,
    ResultFactory,
    to_legacy,
)
from glosa.domain.errors import ReasoningParseError
from glosa.domain.index import DocIndex
from glosa.domain.navigate import NavigateConfig, NavigateStrategy
from glosa.infra.docling.projection import DoclingProjector

if TYPE_CHECKING:
    from glosa.domain.values import Trace
    from glosa.ports.chat import ChatModel
    from glosa.ports.document import DocumentProjector

DEFAULT_CACHE_SIZE = 8


class ParseErrorFactory(Protocol):
    """Anything constructible from a model id and a reason."""

    def __call__(self, *, model_id: str, reason: str) -> Exception: ...


class GlosaReasoningRunner:
    """Answers a question against a stored document.

    Args:
        model: The chat backend. Any `ChatModel` — Ollama, OpenAI-compatible,
            or a test double.
        projector: How to turn a stored payload into a projection. Defaults to
            the bundled Docling projector; pass one built with the host's own
            tree reader so the collapse rules have a single implementation.
        config: Loop tuning. Defaults suit a 30-page report on a local 8B model.
        cache_size: How many parsed documents to keep indexed. Studio asks
            several questions of the same document, and re-parsing it each time
            is pure waste.
        include_furniture: Put running heads and footers in front of the model.
            Off by default — they are repeated on every page and answer nothing.
        result_factory / iteration_factory: Host types to build the reply with.
        parse_error_factory: Host exception raised when the backend cannot
            produce a parseable structured reply.
        annotate_status: Prefix non-answered runs with a one-line marker.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        projector: DocumentProjector | None = None,
        config: NavigateConfig | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        include_furniture: bool = False,
        result_factory: ResultFactory = LegacyResult,
        iteration_factory: IterationFactory = LegacyIteration,
        parse_error_factory: ParseErrorFactory = ReasoningParseError,
        annotate_status: bool = True,
    ) -> None:
        self._model = model
        self._projector: DocumentProjector = projector or DoclingProjector()
        self._config = config or NavigateConfig()
        self._cache: OrderedDict[str, DocIndex] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._include_furniture = include_furniture
        self._result_factory = result_factory
        self._iteration_factory = iteration_factory
        self._parse_error_factory = parse_error_factory
        self._annotate_status = annotate_status

    # -- ReasoningRunner ------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when a backend is wired.

        Deliberately does no I/O: Studio calls this on the request path to
        decide between serving and returning 503. Use `health()` for a probe
        that actually touches the backend.
        """
        return bool(self._model.model_id)

    async def run(
        self,
        *,
        document_json: str,
        query: str,
        model_id: str | None = None,
    ) -> Any:
        """Run the loop and return the host's result type."""
        trace = await self.run_trace(document_json=document_json, query=query, model_id=model_id)
        return to_legacy(
            trace,
            iteration_factory=self._iteration_factory,
            result_factory=self._result_factory,
            annotate_status=self._annotate_status,
        )

    # -- native surface -------------------------------------------------------

    async def run_trace(
        self,
        *,
        document_json: str,
        query: str,
        model_id: str | None = None,
    ) -> Trace:
        """Run the loop and return the full native trace, provenance included."""
        model = self._model if model_id is None else self._model.for_model(model_id)
        index = self._index_for(document_json)
        strategy = NavigateStrategy(model, self._config)
        try:
            return await strategy.run(index, query)
        except ReasoningParseError as exc:
            raise self._parse_error_factory(model_id=exc.model_id, reason=exc.reason) from exc

    async def health(self) -> bool:
        """Probe the backend. Never raises."""
        return await self._model.health()

    async def aclose(self) -> None:
        await self._model.aclose()

    # -- index cache ----------------------------------------------------------

    def _index_for(self, document_json: str) -> DocIndex:
        digest = hashlib.sha256(document_json.encode("utf-8")).hexdigest()
        cached = self._cache.get(digest)
        if cached is not None:
            self._cache.move_to_end(digest)
            return cached

        index = DocIndex(
            self._projector.project(document_json),
            include_furniture=self._include_furniture,
        )
        self._cache[digest] = index
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return index
