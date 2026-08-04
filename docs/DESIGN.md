# glosa — design & implementation plan

> Target integration: [Docling Studio](https://github.com/scub-france/Docling-Studio)
> (`document-parser` backend, hexagonal architecture).
> Replaces / supersedes `infra/docling_agent_reasoning.py`.

---

## 1. Positioning

`docling-agent` is a **document-authoring** agent (write / edit / enrich /
extract) with a chunkless RAG loop added on the side. Studio only uses that
loop, and pays the full price of the package for it: `mellea`, an Ollama-only
backend, a synchronous API, and — as Studio's own code comments record — a
private method (`DoclingRAGAgent._rag_loop`) as its integration point.

`glosa` inverts the priorities. It is a **reading and evidence engine**:

- input: a serialized `DoclingDocument` + a question
- output: an answer, a navigation trace, and span-level evidence for each claim
- constraints: async, cancellable, provider-agnostic, `docling-core`-only deps

Non-goals: writing, editing, enriching documents. Those stay with
`docling-agent`. Structured extraction *is* in scope (phase 5) because it is
the same machinery — locate evidence, then emit typed values — and because
Studio can render it with the bbox overlay it already has.

Name is free on PyPI (`glosa`, checked). Fallback: `docling-glosa`.

---

## 2. The document object

glosa reads **Studio's projection of the document**, not the raw
`DoclingDocument`. This is the load-bearing decision of the whole design.

Studio does not show what docling-core iterates. `infra/docling_tree.py`
collapses two things before anything reaches the graph or the canvas
(Studio issue #197):

* an **InlineGroup** — one `groups[]` entry plus N `texts[]` style runs —
  becomes a single `Paragraph` node with concatenated text and the union of the
  runs' provenances. The style runs land in `skip_refs`;
* a **picture**'s descendants — labels lifted out of a diagram — are dropped,
  while the picture node stays.

So `#/texts/3` can be a perfectly valid ref in the `DoclingDocument` and **not
exist** in Studio's graph. An agent that navigates with docling-core's
`iterate_items` will happily report it, and the overlay will highlight nothing.
Conversely it will never see `#/groups/N`, which *are* the paragraph nodes the
user is looking at.

The same applies to section boundaries. The frontend's
`features/analysis/sectionParenting.ts` scopes a section by walking the **NEXT
chain**: each `SectionHeader` (which is both `title` and `section_header` after
`LABEL_MAP`) becomes the current section, and following nodes belong to it.
There is deliberately no level nesting — an `h2` after an `h1` opens a new
scope rather than nesting inside it. A reader that built sections from a level
stack would read one set of nodes and highlight another.

glosa therefore:

| Concern | Source of truth |
| ------- | --------------- |
| Which nodes exist | `build_collapse_index` → `iter_items` minus `skip_refs` |
| Node identity | `elem::<self_ref>` / `page::<n>`, as `infra/docling_graph.py` builds them |
| Reading order | `dfs_order` — the NEXT chain |
| Section scope | the `sectionParenting.ts` rule |
| Element text | inline text concatenated, table as HTML, picture as caption |
| Provenance | `iter_provs` rows, normalized TOPLEFT as `infra/bbox.py` does |

Two consequences worth stating plainly:

1. **Every ref glosa emits resolves to a node the UI already has.** Pinned by
   `tests/contract/test_studio_section_scoping.py`, which transcribes the
   TypeScript rule and asserts glosa's units match it node for node.
2. **`docling-core` is not a runtime dependency.** glosa reads the serialized
   JSON, so it never needs the library that produced it — including its table
   serializer, which is re-implemented from the cell offsets. Runtime deps are
   `httpx` and `pydantic`. docling-core stays as a *test* dependency, to build
   realistic fixtures.

The collapse rules live behind a `TreeReader` port that mirrors Studio's
`DocumentTreeReader`. Pass Studio's own `DoclingTreeReader` — already wired at
`main.py:321` — and the deployment has exactly one implementation; omit it and
glosa uses a bundled mirror.

One rule stays duplicated, and it is the one that hurts: **section scoping lives
in the frontend**. `sectionParenting.ts` derives it at render time from the NEXT
chain, so every backend consumer — glosa, chunking, ingestion — has to re-derive
it and can drift. If the projection ever carried the owning section on each
element node (a `section_ref` field, or an `IN_SECTION` edge), the frontend
would render it instead of computing it, and the duplication would disappear.
Worth a Studio-side issue.

---

## 3. The integration contract

Studio already defines the port. `glosa` targets it exactly — structurally, not
by importing Studio.

```python
# Docling-Studio: document-parser/domain/ports.py
@runtime_checkable
class ReasoningRunner(Protocol):
    @property
    def is_available(self) -> bool: ...

    async def run(
        self, *, document_json: str, query: str, model_id: str | None = None
    ) -> ReasoningResult: ...
```

with

```python
@dataclass(frozen=True)
class ReasoningIteration:
    iteration: int
    section_ref: str
    reason: str
    section_text_length: int
    can_answer: bool
    response: str


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    iterations: list[ReasoningIteration]
    converged: bool
```

The current adapter builds those with
`ReasoningIteration(**it.model_dump())`. **glosa's legacy trace objects are
pydantic models exposing exactly those six fields**, so that line — and the
whole Vue overlay + Cytoscape trace view downstream — keeps working unchanged.

### Three levels of integration, shipped in order

| Level | Studio change | What it buys |
| ----- | ------------- | ------------ |
| **L1 — drop-in** | swap one class in the DI wire-up (`main.py`) | no `mellea`, no Ollama lock-in, no `_rag_loop`, no `IndexError` |
| **L2 — native trace + streaming** | new optional port method `run_stream`, one SSE route | live trace in the UI, evidence spans → exact bbox highlight |
| **L3 — infra reuse** | pass Studio's existing ports into the runner | reasoning reuses OpenSearch vectors, Neo4j graph, first-class chunks |

L1 must land with **zero** frontend changes. That is the acceptance bar.

---

## 4. What is actually wrong with the current loop

Read from `docling_agent/agent/rag.py` @ `main` and Studio's adapter.

**Correctness**

1. `find_json_dicts(answer)[0]` in `_attempt_answer` raises `IndexError` when
   rejection sampling fails — Studio already has a whole exception type
   (`ReasoningParseError`) and a 502 mapping built around this bug.
2. `_extract_page_summaries` / `_extract_page_keyphrases` `break` out of the
   whole loop after the *first* page found, so page mode gets at most one page
   summary.
3. `_collect_flat_section_text` uses `iterate_items` **tree depth** as the
   heading level. On a flat document, depth is constant, so the "stop at the
   next same-or-higher heading" rule mis-fires.
4. `_select_relevant_documents` decides relevance by substring-matching the
   doc id in free-form LLM prose (`if doc_id in response`). A document named
   `report` matches almost anything.
5. No-heading fallback returns **the entire document markdown as the answer**,
   with `converged=True`. Studio renders that as a successful answer.
6. It navigates docling-core's raw item tree, which is not the tree Studio
   projects (§2). Its `section_ref` can name a node the graph does not contain,
   and its section boundaries are not the ones the UI draws.

**Design**

7. The mellea session is linear and accumulates every section read. "Chunkless"
   stops being cheap after 3 hops: the context holds all visited section texts.
8. The full outline is re-injected into every selection prompt. On a
   300-section document the outline alone can exceed the window.
9. Greedy, strictly sequential, one section per iteration, 2 LLM round-trips
   per hop, `max_iterations=5`. Worst case 10 sequential calls; wall-clock is
   the sum, not the max.
10. `visited` is a hard ban. A section that becomes relevant once you have more
   context can never be re-read.
11. `converged` means "the model self-reported `can_answer`". There is no
    grounding check, no abstention path, no distinction between *answered*,
    *not in this document*, and *budget exhausted*.
12. Provenance stops at `section_ref` + `len(text)`. Studio has bbox
    infrastructure (`infra/bbox.py`, TOPLEFT normalization, per-page overlay)
    and nothing to point it at below section granularity.
13. Multi-document = N independent loops, then "synthesize these strings".
    No shared budget, no cross-document citation.

**Operational**

14. Sync + `rich.Console` printing from library code; Studio offloads to
    `asyncio.to_thread` and cannot cancel, time-box, or count tokens.
15. No caching: outline and summaries are recomputed for every query on a
    document whose `document_json` Studio already has stored.
16. Constructor churn between versions (`model_id=` → `backend=`) on top of a
    private-method call site. Every upstream release is a potential break.
17. No eval harness. There is no number to move.

---

## 5. Architecture

```
glosa/
  types.py              Span, Excerpt, Step, Trace + the legacy 6-field projection
  studio/               ← Studio's document object; ✅ shipped
    tree.py             mirror of infra/docling_tree.py — collapses, dfs_order, labels
    ports.py            TreeReader (mirrors Studio's DocumentTreeReader)
    projection.py       Element / Scope / StudioProjection — node ids, order, scoping
    render.py           element → text: inline concat, table HTML, figure caption
  document/
    index.py            DocIndex — units, excerpts, caches over a projection
    outline.py          budget-aware outline rendering
  retrieve/                                                          ← P2
    lexical.py          BM25 over element text (vendored, no dep)
    vector.py           VectorRetriever port (Studio: EmbeddingService + OpenSearch)
    structure.py        caption / footnote / neighbour expansion
    fuse.py             reciprocal-rank fusion + LLM re-rank
  llm/                  ✅ shipped
    port.py             ChatModel: complete / structured(schema) / stream
    base.py             transport, retries, repair round-trip
    ollama.py openai.py
    schema.py           strict-schema conversion + JSON extraction
  strategy/
    navigate.py         ✅ outline navigation, notes memory, budgets, abstention
    hybrid.py           retrieve-then-read                            ← P2
    router.py           picks a strategy from doc size / structure     ← P2
  verify/                                                            ← P3
    ground.py           sentence → evidence alignment, groundedness score
  runtime/
    budget.py           ✅ step / call / wall-clock budget, deadline
    events.py           typed event stream                            ← P3
    journal.py          prompt/response journal → deterministic replay ← P3
  adapters/             ✅ shipped
    studio.py           GlosaReasoningRunner (implements Studio's port)
    legacy.py           six-field projection of a rich Trace
  server/app.py         standalone SSE sidecar                        ← P3
  cli.py                ✅ glosa ask | map        (+ replay | eval    ← P5)
```


Hexagonal, same discipline as Studio. Two rules hold the design together:

* `document/`, `strategy/` and `adapters/` never import an LLM SDK or an HTTP
  client — everything crosses `llm/port.py`;
* nothing outside `studio/` interprets Docling's raw shape. Structure comes
  from `StudioProjection` or it does not come at all.

### The default loop (`strategy/hybrid.py`)

```
0. index      DocIndex from document_json (cached by content hash)
1. shortlist  BM25 over node texts  →  top-N candidate refs
              ∪ LLM ranking over a budget-fitted outline slice
              fused by RRF, expanded structurally (parent, captions, table refs)
2. read       top-k candidates read CONCURRENTLY (k≈3), one structured call
              each → Note(claim, evidence_span, sufficient: bool)
3. decide     answered / need-more / not-in-document
              need-more → new frontier from the notes' open questions,
              re-visiting allowed at a cost, budget decremented
4. compose    answer from Notes only (never from raw accumulated context)
5. verify     each sentence aligned to ≥1 evidence span; unsupported ones
              flagged or dropped; groundedness score attached
```

Working memory is a list of typed `Note`s, not a growing chat transcript. That
is what keeps the context flat across hops and makes step 4 reproducible.

---

## 6. What we do better — the concrete list

**The same object as the UI**

0. **glosa reads what Studio shows** (§2): same nodes, same ids, same reading
   order, same section scoping. Upstream reads docling-core's raw tree, so its
   `section_ref` can name a node the graph does not have and its sections are
   not the ones the viewer draws. On top of the anchor ref, every step carries
   `node_ids` — the exact set of graph nodes it read — so the overlay never has
   to re-derive membership and disagree with the trace.

**Robustness**

1. **Schema-constrained decoding** — Ollama `format: <json-schema>`, OpenAI
   `response_format: json_schema`, llama.cpp GBNF. The `find_json_dicts[0]`
   failure class disappears. When a backend has no constrained mode, a repair
   prompt runs before erroring, and we still raise a `ReasoningParseError`-shaped
   exception so Studio's 502 path is preserved.
2. **Provider-agnostic** — closes the TODO written into Studio's own
   `LLMProviderType` enum ("today only OLLAMA is realizable"). `LLM_PROVIDER_TYPE`
   becomes a real knob: `ollama | openai | vllm | watsonx | litellm`.
3. **Stable public API** — no private methods, semver, contract tests. Studio
   stops tracking upstream constructor changes.
4. **Deps**: `docling-core` + `httpx` + `pydantic`. No `mellea`, no `torch`.
   Studio's 270 MB remote-conversion image stays 270 MB.

**Quality of answers**

5. **Retrieval prior before LLM navigation.** BM25 costs microseconds and cuts
   the search space before the first token is spent. Fixes the
   "outline-too-big" failure and typically removes 1–3 hops.
6. **Abstention as a first-class outcome.** `status: answered |
   not_in_document | insufficient_evidence | budget_exhausted` instead of a
   boolean `converged` that conflates all four. Studio can render "this
   document does not answer that" honestly.
7. **Grounding verification.** Every answer sentence must align to an evidence
   span; a `groundedness` score (0–1) ships in the trace. This is the single
   biggest credibility win for a product whose selling point is *watching* the
   agent read.
8. **Span-level provenance.** `self_ref` + charspan + page + TOPLEFT bbox per
   claim, computed with the same conventions as Studio's `infra/bbox.py`. The
   viewer can highlight the exact sentence, not the whole section.
9. **Structure-aware reading.** Tables serialized as HTML (not flattened),
   captions and footnotes pulled with their figure, list items kept with their
   parent — all available from `DoclingDocument` and currently ignored outside
   page mode.
10. **Real heading-less handling.** Page/layout segmentation instead of
    returning the whole document as the answer.
11. **Multi-document with a shared budget** and per-claim document attribution,
    instead of N loops plus a synthesis prompt.

**Speed & cost**

12. **Parallel frontier** — k candidates read concurrently. Latency becomes
    ~2 round-trips instead of ~10 sequential ones.
13. **Flat context** — notes-based memory keeps per-call prompts roughly
    constant; token cost grows linearly in hops, not quadratically.
14. **Caching** — `DocIndex`, outline, and per-node summaries cached by
    document hash; Studio queries the same document repeatedly, so the second
    question is materially cheaper.
15. **Budgets & cancellation** — `max_tokens`, `max_steps`, `deadline`;
    `asyncio` cancellation propagates. A user closing the panel actually stops
    the run.

**Observability**

16. **Typed event stream** — `step.started`, `node.read`, `note.added`,
    `answer.delta`, `run.finished`. Ready for the SSE plan already noted in
    Studio's `api/reasoning.py` ("no streaming at this step, see design doc §7").
17. **Deterministic replay** — seeded, journaled prompts/responses;
    `glosa replay run.json` reproduces a trace offline. Directly serves
    Studio's debugging mission.
18. **No stdout from library code** — structured logging + OpenTelemetry spans.
19. **Eval harness** — retrieval hit-rate@k, groundedness, abstention accuracy,
    p50/p95 latency, tokens/answer. Every phase below has a number attached.

---

## 7. Phased plan

### P0 — Scaffolding ✅ done

`uv` project, MIT, `ruff` + `mypy --strict`, `pytest` + `pytest-asyncio`, CI on
3.12/3.13. `tests/contract/test_studio_port.py` re-declares Studio's
`ReasoningRunner` Protocol by transcription and asserts
`isinstance(GlosaReasoningRunner(...), ReasoningRunner)` — conformance without a
dependency in either direction.

### P1 — Drop-in replacement ✅ done

Shipped: `StudioProjection` over Studio's collapsed document object (§2) behind
a `TreeReader` port, `DocIndex` (scopes, page fallback, preamble units, render +
excerpt caches), budget-aware outline rendering, `ChatModel` port with Ollama
and OpenAI-compatible adapters, schema-constrained decoding with a repair
round-trip, `NavigateStrategy` (cheap path for short documents, flat notes
memory, four-valued outcome, recoverable ref selection, step/call/deadline
budgets), `GlosaReasoningRunner` with host-type factories, and a CLI
(`glosa ask` / `glosa map`).

87 tests, `mypy --strict` clean, no `docling-core` at runtime. Integration steps
are in [`INTEGRATION.md`](INTEGRATION.md).

*Verified*: section scoping matches the frontend's `computeSectionParents`
node for node; InlineGroup style runs and picture-internal labels never surface;
every emitted ref resolves to a projected node; the runner satisfies Studio's
protocol; iterations expose exactly the six legacy fields; host factories return
Studio's own dataclasses; parse failures surface as the host's exception; a
document is parsed once across queries.

*Not yet verified* (needs a running Studio + Ollama): the end-to-end round trip
against a real backend, and the Vue overlay rendering a glosa trace.

### P2 — Better loop (3–4 days)

BM25 shortlist + RRF fusion, parallel frontier, multi-document with a shared
budget. Notes memory, budgets, abstention and heading-less segmentation landed
early in P1 — they were cheap and they carry most of the robustness win.

*Done when*: on a 20-question benchmark over 5 Studio-converted PDFs —
≥30 % fewer tokens, ≥40 % lower p95 latency, and ≥1 correct abstention where
the current loop hallucinates. Trace shape unchanged, so still zero frontend
work.

### P3 — Rich trace + streaming (3–4 days)

Evidence spans, bbox resolution, grounding verification, typed events, SSE
sidecar in `server/app.py`.

*Studio-side PR*: add an optional `run_stream` to the `ReasoningRunner`
protocol, one SSE route next to `POST /{doc_id}/reasoning`, and extend
`ReasoningIterationResponse` with optional `evidence[]`. All additive — old
clients keep working.

*Done when*: the frontend highlights the exact supporting region in the PDF
viewer while the run is still in flight.

### P4 — Deep Studio integration (3–4 days)

`VectorRetriever` backed by Studio's `EmbeddingService` + OpenSearch
`VectorStore`; `StructuralNeighbors` backed by the Neo4j `GraphReader`;
citations that carry `chunk_id` so an answer links to first-class chunks.

*Done when*: with ingestion enabled, reasoning uses the existing index instead
of re-scanning, and cited chunks are clickable in the chunk view.

### P5 — Extraction + eval (3–4 days)

`glosa extract --schema invoice.json` returning typed fields *with* evidence
spans; public eval harness and a benchmark page in Studio's docs.

Total ≈ 3–4 weeks of focused work, with something shippable at the end of P1.

---

## 8. Studio-side changes (minimal by design)

| Phase | File | Change |
| ----- | ---- | ------ |
| P1 | `document-parser/main.py` | instantiate `GlosaReasoningRunner` instead of `DoclingAgentReasoningRunner` |
| P1 | `infra/settings.py` | `LLM_PROVIDER_TYPE` gains real values; `REASONING_MODEL_ID` unchanged |
| P1 | `domain/value_objects.py` | `LLMProviderType` gains `OPENAI`, `VLLM`, `WATSONX` |
| P3 | `domain/ports.py` | optional `run_stream` on `ReasoningRunner` |
| P3 | `api/reasoning.py` | `GET /{doc_id}/reasoning/events` (SSE) |
| P3 | `api/schemas.py` | optional `evidence[]` on the iteration response |
| P4 | wire-up | pass `EmbeddingService` / `VectorStore` / `GraphReader` into the runner |

`infra/docling_agent_reasoning.py` stays in the tree as a fallback adapter for
one release, selected by `REASONING_RUNNER=docling-agent|glosa`.

---

## 9. Decisions

| Question | Decision |
| -------- | -------- |
| License | **MIT** |
| Repo layout | **Standalone package** — independently testable, publishable, usable outside Studio |
| Python floor | **3.12** — matches `document-parser`'s `requires-python = ">=3.12"`, so glosa installs into the same venv |
| Dependencies | `docling-core`, `httpx`, `pydantic`. Nothing else, ever, in the core |

Still open:

1. **Benchmark corpus** — ~5 representative PDFs + 20 questions with known
   answers, ideally from a real engagement. Deferred by agreement; it gates
   P2's *acceptance criteria*, not P2's code, so the loop work can start and be
   measured retroactively once the corpus exists.
2. **Upstreaming** — the `IndexError`, the page-summary `break`, and the
   flat-section depth bug are worth PRs to `docling-agent` regardless of what
   Studio ends up running. Cheap, and good citizenship.
