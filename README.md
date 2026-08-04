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

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design and implementation plan.

## Status

Design phase. Nothing is implemented yet.
