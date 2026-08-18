# Wiring glosa into Docling Studio

Phase L1 — drop-in replacement for `infra/docling_agent_reasoning.py`, with no
change to the frontend, the API layer, or the domain ports.

## 1. Dependency

In `document-parser/pyproject.toml`:

```toml
dependencies = [
  # ...
  "glosa>=0.1.0,<1.0.0",
]
```

`docling-agent` and `mellea` can be dropped. glosa adds **nothing** to the
image: its runtime dependencies are `httpx` and `pydantic`, both already in
`document-parser`. It does not even need `docling-core` — it reads the
serialized `document_json`, so the library that produced the document is not
required to read it.

## 2. Wire-up (`document-parser/main.py`)

glosa builds the reply with **Studio's own domain types**, so nothing has to be
translated on the way back:

```python
from glosa import GlosaReasoningRunner, OllamaChatModel, OpenAIChatModel

from domain.ports import ReasoningParseError
from domain.value_objects import LLMProviderType, ReasoningIteration, ReasoningResult


def _build_chat_model(settings):
    if settings.llm_provider_type == LLMProviderType.OLLAMA:
        return OllamaChatModel(
            base_url=settings.ollama_host,
            model_id=settings.reasoning_model_id,
        )
    return OpenAIChatModel(
        base_url=settings.openai_base_url,
        model_id=settings.reasoning_model_id,
        api_key=settings.openai_api_key,
    )


if settings.reasoning_enabled:
    app.state.reasoning_runner = GlosaReasoningRunner(
        _build_chat_model(settings),
        # The DoclingTreeReader already wired at main.py:321 for ChunkService.
        # Reusing it means one implementation of the InlineGroup / picture
        # collapse rules in the whole deployment.
        tree_reader=app.state.tree_reader,
        result_factory=ReasoningResult,
        iteration_factory=ReasoningIteration,
        parse_error_factory=ReasoningParseError,
    )
```

That is the whole change. `api/reasoning.py` keeps its `ReasoningResultResponse`
mapping, its 503 on `is_available`, and its 502 on `ReasoningParseError`.
The `_build_chat_model` helper reads two settings section 3 adds for the
OpenAI-compatible path: `openai_base_url` and `openai_api_key`.

### What can escape `run()`

Three exception classes cross the boundary, each typed and each replaceable
with a host exception via a factory:

| Failure | Default | Factory |
| ------- | ------- | ------- |
| backend cannot satisfy the schema (after constrained decoding + repair) | `parse_error_factory` — Studio's `ReasoningParseError`, mapped to 502 | `parse_error_factory=` |
| backend unreachable / kept failing (`is_available` deliberately does no I/O, so "Ollama is down" surfaces here) | glosa's `BackendError` (carries `status_code` and `retryable`) | `backend_error_factory=` |
| `document_json` is not a readable document | glosa's `DocumentParseError` | `document_error_factory=` |

Anything else indicates a bug in glosa. Transient failures *during* a run that
already produced steps do not raise at all: the run returns a partial trace
with `converged=False` instead of discarding the steps.

Two things that were needed before and are not any more:

- `os.environ["OLLAMA_HOST"] = provider.host` — the host is now an argument, so
  there is no process-wide mutation and no cross-request coupling.
- the `deps_present()` import probe — glosa has no optional heavy dependencies,
  so if it imports, it runs.

## 3. Settings

`LLM_PROVIDER_TYPE` becomes a real switch. `domain/value_objects.py`:

```python
class LLMProviderType(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"  # also vLLM, llama.cpp server, LiteLLM, TGI
```

Everything else (`REASONING_ENABLED`, `OLLAMA_HOST`, `REASONING_MODEL_ID`) keeps
its current meaning.

## 4. Tuning

```python
from glosa import HybridConfig

GlosaReasoningRunner(
    model,
    config=HybridConfig(
        max_steps=6,  # reads before the loop gives up
        max_llm_calls=20,  # hard ceiling on round-trips
        deadline_s=180.0,  # wall-clock ceiling — in-flight calls are cancelled at the deadline
        outline_char_budget=6_000,  # document map size per prompt
        excerpt_char_budget=8_000,  # section text size per read
        direct_char_threshold=6_000,  # below this, read the whole document in one call
        fanout=3,  # candidates read concurrently when the ranking is flat
    ),
)
```

## 5. What the frontend gains for free

Nothing breaks, and four things improve without a line of Vue changing:

- **Every `section_ref` resolves.** glosa navigates the projected document —
  `iter_items` minus `skip_refs`, ordered by `dfs_order` — so a step can never
  name an InlineGroup style run or a picture-internal label, which are refs the
  graph does not contain. It reports `#/groups/N` where the UI shows a
  collapsed paragraph, because that is the node you are looking at.
- **Sections are the ones the UI draws.** Unit boundaries follow
  `sectionParenting.ts`: NEXT chain, one `SectionHeader` to the next, no level
  nesting. Pinned by a contract test that transcribes the TypeScript rule.
- **Honest non-answers.** A run that ends in `not_in_document` /
  `insufficient_evidence` / `budget_exhausted` returns `converged=False` and an
  answer prefixed with a one-line marker, instead of a confident hallucination.
- **No more 502 on a formatting slip.** The schema is pushed into the decoder,
  so `ReasoningParseError` now means the backend genuinely cannot comply.

## 6. Richer trace, when you want it

`run_trace()` is the native surface. Same inputs, but the steps carry page
numbers, TOPLEFT bounding boxes and refs for everything read:

```python
trace = await runner.run_trace(document_json=doc_json, query=q)
for step in trace.steps:
    step.node_ids  # ('elem::#/texts/4', 'elem::#/groups/1', …)
    for span in step.spans:
        span.node_id  # the Cytoscape node — select it, no mapping needed
        span.page_no  # page for the canvas
        span.bbox  # TOPLEFT (l, t, r, b), or None when unknown
```

`step.node_ids` is the set of graph nodes the step actually read. Highlighting
that set is exact by construction — no need to re-run section parenting on the
client and hope it agrees. `bbox` is None rather than Studio's `EMPTY_BBOX`
sentinel when a node has no usable rectangle, so "unknown" stays
distinguishable from "zero-area"; substitute the sentinel at the wire edge.

`trace.status` is the four-valued outcome; `trace.converged` is the boolean
projection of it. This is what phase L2 exposes over SSE.

## 7. If you change the projection

`glosa/infra/docling/tree.py` mirrors `infra/docling_tree.py`. If Studio changes
a collapse rule, a label mapping or the reading order, glosa must follow, and
`tests/contract/test_studio_tree_parity.py` plus
`tests/contract/test_studio_section_scoping.py` are what should fail first.

Passing `tree_reader=app.state.tree_reader` removes half the risk outright: the
collapse rules then come from Studio at runtime, and only the section-scoping
rule stays duplicated — which is itself avoidable if the projection ever carries
the section a node belongs to (see the note at the end of `DESIGN.md` §2).
