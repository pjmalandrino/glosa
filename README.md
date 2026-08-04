# glosa

A grounded document-reading engine for `DoclingDocument`, built to plug into
[Docling Studio](https://github.com/scub-france/Docling-Studio) as a
`ReasoningRunner`.

`glosa` answers questions over a converted document by walking its structure —
no chunking, no external index required — and returns an **auditable trace**:
which nodes it read, why, and where they are on the page.

It navigates the document **as Studio projects it** — the same collapsed nodes,
the same `elem::` ids, the same reading order, the same section boundaries — so
a trace step always resolves to a node the viewer already has.

It is a focused alternative to
[`docling-agent`](https://github.com/docling-project/docling-agent)'s
chunkless RAG loop:

|                        | `docling-agent` v0.1.x            | `glosa`                                           |
| ---------------------- | --------------------------------- | ------------------------------------------------- |
| Document model         | docling-core's raw item tree      | Studio's projection — same nodes, ids, scoping     |
| Public API for Studio  | none (Studio calls `_rag_loop`)   | stable `ReasoningRunner` contract                  |
| LLM backends           | Ollama only (via `mellea`)        | Ollama, OpenAI-compatible, vLLM, watsonx, LiteLLM  |
| Structured output      | ` ```json ` block + regex + retry | schema-constrained decoding, repair fallback       |
| Concurrency            | sync, one section at a time       | async, candidates read in parallel, deadline-bound |
| Retrieval prior        | none (LLM reads the outline)      | BM25 shortlist, heading/body fused by RRF          |
| Provenance             | `section_ref` + char count        | graph node ids + page + TOPLEFT bbox per step      |
| Streaming              | no                                | typed events (SSE-ready)                           |
| Runtime deps           | `mellea`, `docling-agent`         | `httpx`, `pydantic` — not even `docling-core`      |

**Scope.** `glosa` owns *read / answer / cite / extract*. It does not write,
edit or enrich documents — `docling-agent` remains the better tool for that.

## Install

```bash
uv add glosa          # Python 3.12+
```

## Use

```python
from glosa import GlosaReasoningRunner, OllamaChatModel

runner = GlosaReasoningRunner(
    OllamaChatModel(base_url="http://localhost:11434", model_id="granite3.3:8b")
)

trace = await runner.run_trace(document_json=doc_json, query="What is the penalty rate?")
print(trace.status, trace.answer)
for step in trace.steps:
    print(step.ref, step.reason, step.node_ids)  # graph nodes to highlight
```

From the shell:

```bash
glosa map --document analysis.json
glosa ask --document analysis.json --query "What is the penalty rate?"
glosa ask --document analysis.json --query "..." --provider openai --model gpt-4.1-mini
```

Inside Docling Studio, `runner.run(...)` returns Studio's own
`ReasoningResult` / `ReasoningIteration` — see
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the wire-up.

## Layout

Hexagonal: `domain/` (pure logic) depends only on `ports/` (protocols);
`infra/` and `adapters/` implement them. `tests/test_architecture.py` walks the
import graph and fails if a dependency points outward — `json.loads` lives in
`infra/docling` only, `httpx` in `infra/llm` only.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — design, phased plan, what we do differently
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — dropping glosa into Docling Studio

## Status

Alpha. Reading a **single** document is complete (P0–P2): Studio-aligned
projection, BM25 retrieval prior, parallel reads, query-aware excerpt packing,
Ollama and OpenAI-compatible backends, schema-constrained decoding, budgets and
the Studio-facing runner.

Cross-document reading is the next level and is deliberately not started: it
composes single-document reads rather than extending the loop. Sentence-level
evidence and streaming follow — see the plan.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
