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

`docling-agent` and `mellea` can be dropped. glosa's only additions on top of
what Studio already installs are `httpx` and `pydantic` — both already present.

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
        result_factory=ReasoningResult,
        iteration_factory=ReasoningIteration,
        parse_error_factory=ReasoningParseError,
    )
```

That is the whole change. `api/reasoning.py` keeps its `ReasoningResultResponse`
mapping, its 503 on `is_available`, and its 502 on `ReasoningParseError`.

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
from glosa import NavigateConfig

GlosaReasoningRunner(
    model,
    config=NavigateConfig(
        max_steps=6,  # reads before the loop gives up
        max_llm_calls=20,  # hard ceiling on round-trips
        deadline_s=180.0,  # wall-clock ceiling; cancellation is honoured
        outline_char_budget=6_000,  # document map size per prompt
        excerpt_char_budget=8_000,  # section text size per read
        direct_char_threshold=6_000,  # below this, read the whole document in one call
        allow_revisit=True,
    ),
)
```

## 5. What the frontend gains for free

Nothing breaks, and three things improve without a line of Vue changing:

- **Honest non-answers.** A run that ends in `not_in_document` /
  `insufficient_evidence` / `budget_exhausted` returns `converged=False` and an
  answer prefixed with a one-line marker, instead of a confident hallucination.
- **Real section boundaries on flat documents.** Most converted PDFs are flat;
  the section a trace step points at is now the actual section.
- **No more 502 on a formatting slip.** The schema is pushed into the decoder,
  so `ReasoningParseError` now means the backend genuinely cannot comply.

## 6. Richer trace, when you want it

`run_trace()` is the native surface. Same inputs, but the steps carry page
numbers, TOPLEFT bounding boxes and refs for everything read:

```python
trace = await runner.run_trace(document_json=doc_json, query=q)
for step in trace.steps:
    for span in step.spans:
        span.self_ref, span.page_no, span.bbox  # ready for the canvas overlay
```

`trace.status` is the four-valued outcome; `trace.converged` is the boolean
projection of it. This is what phase L2 exposes over SSE.
