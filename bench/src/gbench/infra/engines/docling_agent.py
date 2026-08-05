"""`docling-agent`'s `DoclingRAGAgent` as a bench engine.  **B2 — not yet run.**

Behind the `[docling-agent]` extra and imported inside `__init__`, so nothing
here is loaded — and neither is `mellea` — unless the engine is selected.

Three impedance mismatches, all of them properties of the thing being measured
rather than of the harness:

* it is **synchronous** and prints to stdout from library code, so the call goes
  through `asyncio.to_thread` exactly as Docling Studio does today;
* it takes a **`DoclingDocument` object**, not the serialized JSON, so the
  adapter deserializes the corpus's committed file with `docling-core`. Same
  bytes as glosa reads (`EVAL.md` §3.1);
* its trace names **sections of docling-core's raw item tree**, which is not the
  tree the corpus's `gold_refs` are authored against (`DESIGN.md` §2). Refs that
  do not resolve in the projection are kept verbatim in the journal and simply
  miss on `hit@k`. That is a real difference between the engines, so it must not
  be papered over by fuzzy matching — but it does mean this engine's `hit@k` is
  a lower bound, and the report says so.

The upstream trace shape (`RAGTrace` / `RAGResult`) is read defensively: this
adapter is pinned by a contract test at B2, against an installed version, and
not before. Until that test exists the field names below are transcription from
the published signature, not verified fact.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from gbench.ports.engine import BenchDocument, Capabilities, EngineAnswer, PrepareCost

if TYPE_CHECKING:
    from gbench.domain.item import RenderedItem


class DoclingAgentEngine:
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        max_iterations: int = 5,
        top_k: int = 10,
    ) -> None:
        try:
            from docling_agent.agent.rag import DoclingRAGAgent
            from docling_agent.backend import BackendConfig, ModelConfig, create_backend
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise RuntimeError(
                "the docling-agent engine needs `uv sync --extra docling-agent`"
            ) from exc

        self._backend = create_backend(
            BackendConfig(
                type="ollama",
                base_url=base_url,
                models=ModelConfig(reasoning=model_id, writing=model_id),
            )
        )
        self._agent = DoclingRAGAgent(
            tools=[],
            backend=self._backend,
            max_iterations=max_iterations,
            top_k=top_k,
            verbose=False,
        )
        self._documents: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "docling-agent"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            read_refs=True,
            abstention=True,
            llm_calls=False,
            groundedness=False,
            abstention_signal="converged == False",
        )

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        """Deserialize once. No LLM call — it navigates headings, like glosa."""
        started = time.monotonic()
        cached = doc.slug in self._documents
        if not cached:
            from docling_core.types.doc import DoclingDocument

            self._documents[doc.slug] = DoclingDocument.model_validate_json(doc.docling_json)
        return PrepareCost(
            llm_calls=0,
            prompt_chars=0,
            wall_s=round(time.monotonic() - started, 3),
            cached=cached,
        )

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        await self.prepare(doc)
        document = self._documents[doc.slug]

        started = time.monotonic()
        try:
            trace = await asyncio.to_thread(
                self._agent.run_with_trace, task=item.prompt, document=document
            )
        except Exception as exc:
            return EngineAnswer(
                text="",
                error=f"{type(exc).__name__}: {exc}",
                wall_s=round(time.monotonic() - started, 3),
            )

        result = _first_result(trace)
        iterations = list(getattr(result, "iterations", ()) or ())
        converged = bool(getattr(result, "converged", False))
        return EngineAnswer(
            text=str(getattr(trace, "final_answer", "") or getattr(result, "answer", "")),
            abstained=not converged,
            read_refs=tuple(
                str(getattr(it, "section_ref", ""))
                for it in iterations
                if getattr(it, "section_ref", "")
            ),
            llm_calls=None,  # not reported upstream; ≈ 2 per iteration by construction
            prompt_chars=sum(int(getattr(it, "section_text_length", 0) or 0) for it in iterations),
            wall_s=round(time.monotonic() - started, 3),
            trace={"converged": converged, "iterations": len(iterations)},
        )

    async def aclose(self) -> None:
        return None


def _first_result(trace: Any) -> Any:
    """`RAGTrace` carries one `RAGResult` per document; the bench runs one."""
    results = getattr(trace, "results", None) or getattr(trace, "per_document", None)
    if not results:
        return trace
    if isinstance(results, dict):
        return next(iter(results.values()))
    return results[0]
