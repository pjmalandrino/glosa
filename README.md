# glosa

A grounded document-reading engine for `DoclingDocument`, built to plug into
[Docling Studio](https://github.com/scub-france/Docling-Studio) as a
`ReasoningRunner`.

`glosa` answers questions over a converted document by walking its structure —
no chunking, no external index required — and returns an **auditable trace**:
which nodes it read, why, and which spans support each sentence of the answer.

It is a focused alternative to
[`docling-agent`](https://github.com/docling-project/docling-agent)'s
chunkless RAG loop:

|                        | `docling-agent` v0.1.x            | `glosa`                                           |
| ---------------------- | --------------------------------- | ------------------------------------------------- |
| Public API for Studio  | none (Studio calls `_rag_loop`)   | stable `ReasoningRunner` contract                  |
| LLM backends           | Ollama only (via `mellea`)        | Ollama, OpenAI-compatible, vLLM, watsonx, LiteLLM  |
| Structured output      | ` ```json ` block + regex + retry | schema-constrained decoding, repair fallback       |
| Concurrency            | sync, blocking                    | async-native, cancellable, deadline-bounded        |
| Retrieval prior        | none (LLM reads the outline)      | BM25 + optional vectors, fused with LLM ranking    |
| Provenance             | `section_ref` + char count        | `self_ref` + charspan + page + bbox per claim      |
| Streaming              | no                                | typed events (SSE-ready)                           |
| Heavy deps             | `mellea`, `docling-agent`         | `docling-core` only                                |

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
    print(step.ref, step.reason, [s.bbox for s in step.spans])
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

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — design, phased plan, what we do differently
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — dropping glosa into Docling Studio

## Status

Alpha. Phases P0–P1 are done: indexing, outline navigation, Ollama and
OpenAI-compatible backends, schema-constrained decoding, budgets, and the
Studio-facing runner. Retrieval pre-ranking, parallel reads, evidence spans and
streaming are next — see the plan.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
