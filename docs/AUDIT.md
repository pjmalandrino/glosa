# glosa — product audit

> Date: 2026-08-18 · Tree audited: `f358511` · Suite state: 210 tests pass,
> `mypy --strict` clean, `ruff check` / `format` clean.
>
> Method: a full read of `src/`, `tests/` and `docs/`, then seven parallel
> audit passes (reading loop, index/projection, retrieval, LLM infra,
> integration/API/packaging, test quality, product strategy). Every finding
> labelled **[confirmed]** below was re-verified adversarially by reproducing
> the failure against the current tree — the reproduction is quoted in the
> finding. Findings without the label are traced in the code but not executed.

---

## 1. Verdict

glosa is an unusually well-engineered alpha. The three ideas it is built on are
sound and real in the code: **read the projection the viewer draws** (so every
ref in a trace resolves), **retrieval proposes / the model confirms** (so the
model spends calls reading, not choosing), and **an auditable, four-valued,
quote-checked trace** (so a non-answer is honest and an answer is checkable).
The hexagonal layering is enforced by an executable test, the runtime footprint
really is `httpx` + `pydantic`, and the docs mostly practice the honesty they
preach.

The audit found no flaw in those ideas. It found that the *guarantees built on
them stop short of where the docs say they reach*, in five clusters:

1. **The parity promise breaks on nested documents.** `scopes()` ignores the
   explicit-parent exemption that the frontend rule it transcribes applies, so
   on a document where a heading has trailing content after a nested
   sub-heading, glosa reads one set of nodes and the UI draws another — the
   exact failure class the product exists to prevent (§3, B1).
2. **The resilience story is thinner than claimed.** One transient backend
   error in the wrong place destroys a whole accumulated trace; the wall-clock
   deadline is advisory (a run can overshoot it several-fold); non-retryable
   4xx responses are retried; malformed input crashes with raw
   `RecursionError` instead of the promised typed error (§3–§4).
3. **The lexical prior underperforms on exactly the target corpus.** The
   inflection folder splits core French singular/plural pairs into different
   buckets, stopwords flood coverage and heading ranks, the RRF-ratio margin
   cannot open the gate on "Article 7"-style documents (its motivating case),
   and non-Latin documents silently lose the prior and the leads entirely (§5).
4. **A handful of ✅ claims are ahead of the code** — span-level provenance,
   OpenTelemetry, watsonx, streaming, `uv add glosa` — and the canonical
   `INTEGRATION.md` wire-up crashes with a `TypeError` as written (§7).
5. **The product has no number.** No benchmark corpus, no eval harness, no
   token accounting, no end-to-end run against a real Studio or a real model —
   while the niche competitor (PageIndex) leads with 98.7% on FinanceBench.
   Every quality claim is currently pinned only against scripted fakes (§8).

None of this is architectural. Each cluster has a bounded fix, and the
roadmap in §10 orders them.

---

## 2. What is genuinely solid

Recorded so the rest of this document is read in proportion.

- **The trust → expand → hedge escalation** (`domain/hybrid.py`) is coherent,
  pinned by tests, and honestly reasoned in code comments. The
  expansion-weight bound (a guessed term can never outrank a user-term match
  *within the fused ordering*) is a real arithmetic argument with a pinned
  test.
- **The grounded-quote check** costs +135 prompt chars and no extra call, and
  its blind spot is documented rather than patched. No competitor ships any
  citation check.
- **The layering is executable.** `tests/test_architecture.py` walks the
  import graph; `json.loads` lives in one package, `httpx` in one package; the
  domain runs against a hand-written non-Docling projection in tests.
- **Long-document mechanics exist end to end**: budget-fitted outline that
  returns the refs it actually showed, coarse-to-fine descent into elided
  children, query-aware packing, retrieval indexing full section text (the
  past excerpt-indexing bug is genuinely fixed and regression-pinned —
  `tests/test_packing.py:142`).
- **Wire compatibility is credible**: host-type factories, six legacy fields,
  status prefixes so a UI that renders only `answer` still tells the truth.
- **Projection collapse rules** (InlineGroup merge, picture-descendant drop,
  orphan re-attachment, TOPLEFT flip) are correct on everything the fixtures
  reach.

---

## 3. Confirmed correctness bugs

Ordered by product impact. "Scenario" is the reproduced failure.

### B1 — Section scoping diverges from the frontend rule on nested documents *[confirmed]*

`src/glosa/infra/docling/projection.py:204` (`scopes()`)

- **Scenario.** `h1 → [intro, h2 → [body], trailing text parented to h1]`,
  built with docling-core. DFS order puts the trailing text after the `h2`, so
  `scopes()` attributes it to the `h2`. The oracle transcribed from the
  frontend (`tests/contract/test_studio_section_scoping.py`) applies the
  explicit-PARENT_OF exemption (`elif current_section and node not in
  explicit_parent_of`) and attributes it to the `h1`. Reproduced:
  `_assert_scoping_matches` fails with
  `(elem::#/texts/4 → elem::#/texts/0 vs elem::#/texts/2)`.
- **Impact.** A step reading the `h2` reports `node_ids` the UI draws inside
  the `h1`; a step reading the `h1` omits its trailing text — if the answer
  lives there, the read comes back empty. This breaks DESIGN §2/§6.0's core
  promise ("scoping matches the frontend node for node") for nested documents,
  which `conftest.py` itself calls "the well-formed case" (DOCX/HTML
  conversions). The suite stays green only because no fixture places content
  under a heading after a nested sub-heading.
- **Fix.** In `scopes()`, consult `element.parent` before NEXT-chain
  attribution: if it resolves to a projected `SectionHeader`, attribute the
  element there. Add the discriminating fixture — the oracle already catches
  it.

### B2 — Query-aware packing drops the answer-bearing passage when it alone exceeds the budget *[confirmed]*

`src/glosa/domain/index.py:389` (`_assemble_by_relevance`)

- **Scenario.** A section whose top-BM25 element (the one carrying the answer)
  is larger than the remaining budget: the loop does
  `if used + len(element.text) > char_budget: continue` — no truncation — and
  packs smaller zero-score fillers instead. Reproduced twice independently:
  budget 8 000 / 10.8k-char answer paragraph → excerpt contains **only**
  non-matching text; budget 600 / 662-char answer paragraph (BM25 4.7 vs 0.95)
  → same. The appended marker then claims "*the ones matching the question are
  shown*", which is false.
- **Impact.** The exact "40-page appendix" case the feature advertises is its
  worst case; head-first truncation (which keeps a partial head) would have
  done better whenever the answer is early in the big paragraph. The model
  reads a "focused" excerpt without the answer and concludes
  `not_in_document`/`insufficient` — with a trace that looks diligent.
- **Fix.** Truncate the top-ranked element to the remaining budget (as
  `_assemble` does for heads, with the marker); never let a zero-score element
  displace a scored one; only print the "matching passages shown" marker when
  every kept part actually scored.

### B3 — Malformed `document_json` escapes the typed-error contract *[confirmed]*

`src/glosa/infra/docling/tree.py:172` (`dfs_order`), `projection.py:86`

- **Scenario.** `ports/document.py` promises "*Raises DocumentParseError if
  the payload is not a document*", but only `json.loads` + `isinstance(dict)`
  is guarded. Reproduced, all escaping raw: a `children` cycle →
  `RecursionError` (`dfs_order.walk` has no visited set — the collapse helpers
  do, the order walk does not); `{"texts": "oops"}` / int items →
  `AttributeError`; non-numeric bbox / charspan / `num_rows` → `ValueError`;
  non-dict page entry → `AttributeError`. Also: a ref listed twice as a child
  is **silently projected twice** (`elements` = `['#/texts/0', '#/texts/0']`),
  doubling its text in scopes, `char_len` and excerpts.
- **Impact.** A host mapping `DocumentParseError` to a 4xx gets an unhandled
  500 on corrupt or adversarial stored JSON. Compounding: `studio.py`'s
  `_index_for` runs *outside* the `try` in `run_trace`, so even a properly
  raised `DocumentParseError` crosses the boundary untranslated (see R4).
- **Fix.** Visited-set in `dfs_order` (fixes cycle + duplicates), defensive
  coercion in `iter_provs` / `iter_pages` / `render_table` (reuse
  `_as_float`/`_as_int`), and wrap `DoclingProjection._build` so residual
  `TypeError`/`ValueError`/`AttributeError`/`RecursionError` re-raise as
  `DocumentParseError`.

### B4 — One transient backend failure in the wrong place destroys the whole trace *[confirmed]*

`src/glosa/domain/hybrid.py:237` (`_loop`), `:491` (`_read_batch`), `:294` (`_next_picks`)

- **Scenario.** `_loop` catches only `BudgetExhausted`; the Studio adapter
  translates only `ReasoningParseError`. Reproduced, three ways: (1) after
  three successful grounded rounds, a `BackendError` from **compose** discards
  all three steps; (2) a **decisive shortlist** narrows the round to one read
  (`_fanout_for` → width 1 — the advertised fast path); that single read
  failing makes `_read_batch` re-raise and the run crash, earlier steps lost;
  (3) a failure in the **optional expansion call** (`expand_query`) aborts the
  run the same way, even though `expand_query=False` is a supported config
  whose behaviour is simply "go to the hedge".
- **Impact.** Contradicts the package's own philosophy ("a partial answer is
  worth having; a crash on the way to it is not" — pinned in tests, but only
  for the budget case) and the auditable-trace selling point: the audit trail
  is exactly what gets thrown away.
- **Fix.** Compose failure → fall back to notes; expansion failure → log, mark
  attempted, proceed to hedge; whole-round read failure after ≥1 successful
  round → return the partial trace with a status carrying the error. Reserve
  raw propagation for `ReasoningParseError` (the 502 contract needs it).

### B5 — The wall-clock deadline does not bind *[confirmed]*

`src/glosa/domain/hybrid.py:171`, `src/glosa/infra/llm/base.py:151`, `src/glosa/cli.py:63–77`

- **Scenario.** `check_deadline()` runs once per round; no `asyncio.timeout`
  anywhere; `Budget.remaining_s` is never plumbed into the transport.
  Reproduced: `deadline_s=0.5` with 3-second calls → run returned after
  **12.0 s and 5 model calls** (24× the deadline). Worst case with defaults:
  one `structured()` call = 2 chats × 3 attempts × 120 s ≈ **723 s inside a
  180 s deadline**. The CLI passes the *same* `--timeout` as both the HTTP
  per-request timeout and `deadline_s`, so one hung request legally consumes
  the whole budget before the first check. The cheap path
  (`read_whole_document`) never checks the deadline at all.
- **Impact.** README sells "deadline-bound"; `budget.py`'s stated purpose is
  embedding in a request handler. A stalled Ollama makes Studio's handler hang
  for minutes. Related under-accounting: `spend_call()` charges 1 while
  `structured()` can issue up to 6 HTTP requests.
- **Fix.** Clamp each request's timeout to `min(configured,
  budget.remaining_s)` and skip retries that cannot fit; or wrap each round in
  `asyncio.timeout(budget.remaining_s)`. In the CLI, split `--timeout` into
  `--deadline` and `--request-timeout`.

### B6 — `_RETRYABLE_STATUS` is dead code: every 4xx is retried like a 503 *[confirmed]*

`src/glosa/infra/llm/base.py:154`

- **Scenario.** Both the retryable branch and the generic `>= 400` branch
  raise the same `BackendError`, and the same `except` clause catches and
  retries both. Reproduced with `MockTransport`: 401, 404 and 400 each
  produced **exactly 3 identical requests** plus 1.5 s of backoff; a 200 body
  with no content is likewise re-sent 3 times. The curated frozenset is
  referenced nowhere else.
- **Impact.** Deterministic failures (bad key, missing model, schema-rejected
  payload) surface late and in triplicate against paid APIs; against a
  rate-limiting gateway the pointless retries extend a lockout. The docstring
  ("Retries on transient transport/5xx failures") describes code that does not
  exist.
- **Fix.** A non-retryable error class (or `retryable: bool` on
  `BackendError`, with `status_code`) excluded from the retry `except`; tests
  asserting 401/404 → exactly one request. This also unblocks the error
  taxonomy (R3).

### B7 — `fold()` applies English rules to French words, splitting the pairs it exists to join *[confirmed]*

`src/glosa/domain/lexical.py:63–67`

- **Scenario.** Reproduced with the shipped function: `partie → partie` but
  `parties → party`; `garantie → garantie` but `garanties → garanty`;
  `bureau → bureau` but `bureaux → bureal`; same for
  `réseau(x)`, `niveau(x)`, `tableau(x)`, `eau(x)` (the `-aux` rule fires
  before the generic `-x` strip that would have been correct). Verified
  consequence: query "durée de la **garantie**" scores **zero content hits**
  against a document that says "les **garanties** expirent…" — and because
  `garantie` then has df = 0 it is excluded from the *answerable* set, so the
  coverage gate cannot even notice the miss.
- **Impact.** This is precisely the singular/plural failure class the
  docstring says `fold()` removes ("livraisons vs livraison"), on the
  product's stated target corpus. `tests/test_lexical.py` pins only the pairs
  that work.
- **Fix.** Strip a trailing `s` *before* the `-x`/`-aux` collapses
  (`bureaux → bureau`); fold `-ie`/`-ies` to one bucket; parametrized tests
  for `partie(s)`, `garantie(s)`, `bureau(x)`, `réseau(x)`, `eau(x)`.

### B8 — Non-Latin documents silently lose the prior, and the lead pruner deletes every lead *[confirmed]*

`src/glosa/domain/lexical.py:45`, `src/glosa/domain/index.py:163`

- **Scenario.** `_TOKEN = [0-9a-z]+` after NFKD casefold. Reproduced:
  `tokenize(<Japanese sentence>) == []`; Cyrillic keeps only digits. Three
  fully distinct Japanese leads → `_prune_boilerplate_leads` sees empty
  `rare_terms` for each → **every lead replaced by ""** (the distinctiveness
  test reads "no evidence of distinction" as "boilerplate"). `shortlist()` is
  always empty, so every question silently pays full model navigation; nothing
  reports the degradation.
- **Impact.** Not the target market today, but the failure is silent and the
  "1 call instead of 4" claim inverts wholesale on such documents.
- **Fix.** Unicode-aware token pattern (or CJK bigrams); at minimum,
  `_prune_boilerplate_leads` must keep a lead whose tokenization is empty.

### B9 — Tables above 4 000 grid positions lose structure and silently lose cells *[confirmed]*

`src/glosa/infra/docling/render.py:86, 139`

- **Scenario.** The gate is on `num_rows * num_cols` (grid area, not cell
  count): a 500×9 table (4 500 cells) — or a sparse table whose *declared*
  grid is large — trips the rows-only fallback. Reproduced: output has no
  `<tr>`/newlines (cells joined with `" | "`), and `cells[:4000]` drops the
  last ~55 rows — where totals rows live — while the marker says nothing about
  omission.
- **Impact.** Destroys exactly the row/column association the module docstring
  gives as the reason tables are HTML-serialized ("what a numeric question
  needs"), and does it silently. Adjacent finding: overlapping cells and cells
  outside the declared grid are silently dropped in the normal path too
  (`render.py:110` — `covered` is consulted before `placed`).
- **Fix.** Gate on `len(cells)`; group the fallback by
  `start_row_offset_idx` (already on every cell) so rows survive; extend the
  marker with the omitted count; expand the grid to max cell extents rather
  than dropping out-of-grid cells.

### B10 — The grounding fold misses curly apostrophes and ligatures: false «ungrounded» on genuine French quotes *[confirmed]*

`src/glosa/domain/reading.py:143`, `src/glosa/domain/lexical.py:41`

- **Scenario.** NFKD does not decompose U+2019 (’) or œ/æ. Reproduced:
  document "Le plafond d’indemnisation s’élève…" (curly, as word-processor
  PDFs are), model quote "Le plafond d'indemnisation…" (ASCII, as models
  type) → substring test fails → `grounded=False` plus a "claims a quote
  absent from the text" warning, on a quote that is really there. Same for
  `œuvre` vs `oeuvre` — which also splits *retrieval* tokens
  (`maître d'œuvre` tokenizes to `['maitre', 'uvre']`).
- **Impact.** Systematic false negatives on French corpora poison the exact
  signal DESIGN §6.7 says will produce the calibration labels, and the CLI
  renders a scary warning for correct answers.
- **Fix.** A small translation table before the substring test (and in
  `normalize()`): `’ ‘ ʼ → '`, `“ ” → "`, `œ → oe`, `æ → ae`, dash variants.

### Smaller confirmed defects

| | Where | Defect |
|---|---|---|
| B11 | `hybrid.py:432` | `gap_notes=0` feeds **all** notes into the probe (`notes[-0:]` is the whole list) — the exact opposite of disabling the feature. *[confirmed]* |
| B12 | `hybrid.py:392`, `reading.py:266` | Hedge + descent can pick the lexical top **twice in one round** (descent candidates are not filtered against the round's picks): two steps, two calls, same excerpt. |
| B13 | `hybrid.py:386` | The hedge step's recorded reason claims "the model's own pick is read alongside" even when budget truncated the hedge and **no model was consulted** — the audit trail asserts an event that did not occur. *[confirmed]* |
| B14 | `hybrid.py:351` | An **empty** expansion is re-requested and re-billed every low-confidence round ("once per run" is tracked by tuple truthiness, not an attempted flag). *[confirmed]* |
| B15 | `hybrid.py:137`, `reading.py:362` | Cheap path raises `BudgetExhausted` to the caller where the loop path returns a `BUDGET_EXHAUSTED` trace (`max_steps=0`/`max_llm_calls=0`). *[confirmed]* |
| B16 | `projection.py:181`, `index.py:110` | `has_sections` counts **furniture** headings that `scopes()` then excludes: a document whose only heading is a running header skips the page fallback and collapses into one mis-titled pseudo-section. |
| B17 | `base.py:158` | A 200 with a non-JSON body (misconfigured proxy) escapes as raw `json.JSONDecodeError`; a non-dict `choices[0]` as `AttributeError` — outside the typed hierarchy. |
| B18 | `base.py:114` | `health()` violates its "must not raise" contract: a `for_model` twin probed after the owner's `aclose()` raises `RuntimeError` (closed shared client). |
| B19 | `cli.py:84` | `--json` — the mode built for scripting — always exits 0; the human mode exits 1 on any non-answered status. Crash exits are indistinguishable from "not converged". |
| B20 | `hybrid.py:484` | A pick whose read fails is never marked visited: a deterministically failing unit stays top-ranked and is re-picked every round, burning a step + call per round until the budget dies. |

---

## 4. Resilience & robustness — the shape of the problem

The individual defects above share one root: **error handling and budgets were
built for the happy path plus `BudgetExhausted`, and every other failure mode
falls through.** Concretely:

- **The boundary maps one error.** `run()`/`run_trace()` translate
  `ReasoningParseError` only. `BackendError` (the routine "Ollama is down"
  case — `is_available` deliberately does no I/O, so it stays `True`) and
  `DocumentParseError` (raised in `_index_for`, *outside* the `try`) cross
  into Studio as unmapped internal exceptions → opaque 500s.
  `INTEGRATION.md` documents Studio's handling as "503 on `is_available`, 502
  on `ReasoningParseError`" — there is no row for the common failure.
  → Add `backend_error_factory` / document-error mapping hooks, and document
  which exceptions can escape `run()`.
- **`BackendError` is a message string.** No status code, no retryability, no
  provider. A host cannot distinguish 429 (back-pressure) from 401
  (misconfiguration) from 5xx (incident) without parsing prose; `Retry-After`
  is ignored; backoff is fixed and jitterless.
- **Token accounting does not exist.** Ollama's `eval_count` and OpenAI's
  `usage` are discarded in `_extract_text`; request ids and rate-limit headers
  are never read. A product whose selling point is an auditable trace cannot
  say what a run cost. (DESIGN's "≥30 % fewer tokens" acceptance criterion is
  currently *unmeasurable by the code itself*.)
- **A design note on `absent_votes`.** `absent` is defined as a
  document-level judgment ("this document is not about the subject") but asked
  per section read; two votes end the run as `NOT_IN_DOCUMENT`. With
  `fanout=3`, one round of two off-topic sections can abstain the whole run
  while the right section is unread. Unverified against real models (no eval
  corpus) — flagging it as a calibration risk to measure, not a confirmed
  bug.
- **OpenAI payload portability.** `temperature` is always sent (the current
  OpenAI reasoning family rejects it → every call 400s, ×3 retries via B6),
  and only `max_completion_tokens` is sent (pre-2024 OpenAI-compatible servers
  honor only `max_tokens` and silently run unbounded).

---

## 5. The lexical prior on the target market

The prior's *architecture* (propose → confirm, never decide alone) is right.
Its *calibration* undermines it on French documents, which are the product's
motivating corpus:

- **No stopword handling anywhere.** «le/de/est/du» are indexed, count as
  *answerable* and *covered* terms, and rank in the heading view. Three
  verified consequences: the documented "empty shortlist = no signal" state
  never occurs for natural questions (every section matches something);
  coverage carries a stopword floor, so `min_coverage=0.34` is nearly always
  met (reproduced: coverage **1.0 on the wrong section** for a typical French
  question); and a heading-view rank-1 on «du» weighs exactly as much in RRF
  as a contentful body rank-1 — which is the mechanism behind the
  expansion-poisoning case below. → A 50–100-word fr/en stopword set (or
  IDF-weighted rank contributions) is the single highest-leverage retrieval
  fix.
- **The margin cannot see magnitude.** `margin` is a ratio of fused RRF
  reciprocals: adjacent ranks in the same views give 62/61 ≈ **1.016
  regardless of the BM25 gap**. Reproduced: a near-4× lexical blowout
  (8.97 vs 2.36) on an "Article 1/7/9" document → margin 1.016 <
  min_margin 1.15 → not confident. On exactly the numbered-heading documents
  the prior was built for, the 1-call path requires multi-view corroboration
  that uninformative headings cannot give — so every query pays the expansion
  and usually the hedge. The docstring "around 1.0 the ranking is flat" is
  untrue of the metric as computed. → Gate on the body view's raw BM25 ratio
  (or view-agreement count), not the fused ratio.
- **The gate saturates on minimal evidence.** One incidental term in one
  section → single candidate → `margin = inf`, `coverage = 1.0` → confident;
  retrieval decides alone on the thinnest possible evidence, contradicting
  "it never decides alone" (reproduced with an insurance-vocabulary query
  against a contract that mentions «sinistre» once). → Require a minimum
  number of matched content terms (or an absolute score floor) before
  trusting; route single-hit shortlists to the hedge.
- **A bad expansion can *manufacture* confidence.** The documented "can only
  add, never displace" bound holds for the fused ordering but not for the
  gate: expansion terms change margin and coverage. Reproduced: a plausible
  register expansion lifts the *wrong* article from margin 1.016 (untrusted,
  harmless) to margin 1.27 / coverage 0.75 → **confident**, demoting the
  correct section to third. → Compute `confident()` over original views only;
  use the expansion to add hedge candidates.
- **Numbers are locale-split.** `2,5` vs `2.5` and `500 000` vs `500000`
  never match — the tokenizer keeps decimal groups but not across locale
  conventions, on a French/English-mixed user base asking numeric questions.
- **Table markup pollutes the index.** `<table>`, `tr`, `th`, `td` are
  indexed as body tokens; a 100-cell table adds ~200 markup tokens of length
  normalization against table-heavy sections — the sections numeric questions
  need. → Strip tags for the lexical view only.

(Plus B7, B8 and B10 above, which are outright bugs in the same layer.)

---

## 6. Fidelity edge cases in the projection

Beyond B1/B3/B9/B16:

- **Prov-less elements are unreachable in page mode.** Page units select by
  `page_no in e.pages`, and pages derive only from prov rows; an element with
  no provenance belongs to no page unit — not in any excerpt, any BM25 entry,
  any `element_refs`. Section mode is immune (reading order, not provenance).
- **Page-mode titles embed raw markup.** Page titles are built from the first
  three elements' raw text with no `is_prose` filter, so a page opening on a
  table is titled `<table><caption>…` — the exact outcome the lead logic
  documents avoiding for sections.
- **The per-focus excerpt cache is unbounded.** `(ref, budget, focus)`-keyed
  entries accumulate per distinct question on up to 8 process-lifetime cached
  indexes. A slow leak in a long-lived server; bound it or stop caching
  focus-keyed entries (assembly is cheap next to the LLM call it feeds).
- **The hard visited ban survives** — the exact flaw DESIGN §4.10 charges the
  predecessor with. `Step.revisited` exists and is never set. It compounds
  with B2: a large section read under the wrong focus is permanently
  exhausted even when later notes reveal it held the answer. Worse, once
  every query term appears somewhere in *visited full text* (not the shown
  excerpt), `_probe` regime 3 stops searching the question's own vocabulary
  entirely. → Allow one bounded re-read (different focus) when the gap
  vocabulary matches text the excerpt omitted; set `revisited=True`.

---

## 7. Claims vs code

The ✅/◐/◻︎ discipline is real and mostly honest — which makes the exceptions
cheap to fix and expensive to leave. An evaluating adopter finds each of these
with grep.

| Claim | Reality | Action |
|---|---|---|
| README: `uv add glosa` / INTEGRATION: `glosa>=0.1.0,<1.0.0` | Not on PyPI (404 as of 2026-08-18); version `0.1.0.dev0` would not satisfy the pin even once published; no build/publish step in CI, so the wheel has never been exercised | Publish 0.1.0 (trusted publishing); add `uv build` to CI; single-source the version |
| INTEGRATION.md §2 wire-up (the canonical example) | **Crashes**: `GlosaReasoningRunner` has no `tree_reader` kwarg (correct form is `projector=DoclingProjector(tree_reader=…)`, shown only in a docstring). Also references `settings.openai_*` never introduced, and §7 points at `glosa/studio/tree.py` (real path: `src/glosa/infra/docling/tree.py`) | Add a `tree_reader` convenience kwarg or fix the doc; a construct-only doctest would pin it |
| README table: providers "Ollama, OpenAI-compatible, vLLM, watsonx, LiteLLM"; DESIGN §6.2 ✅ | Two adapters. vLLM/LiteLLM/TGI legitimately ride the OpenAI adapter; **watsonx cannot** (IAM token exchange, `project_id`, versioned paths — not a static-Bearer `/chat/completions`). CLI: `choices=("ollama", "openai")` | "watsonx via a LiteLLM/OpenAI-compatible gateway", or ship an adapter |
| README table: "Streaming: typed events (SSE-ready)"; DESIGN module map "`chat.py` … / stream" | No stream method on the port, no event type, no SSE anywhere; both adapters pin `"stream": False`. §6.16 honestly says ◻︎ two pages later | Move the README row to "planned (P3)"; fix the module map |
| DESIGN §6.8 ✅ "Span-level provenance … charspan … per claim" | `Span.char_start/char_end` declared, never populated; spans are per-*step* element lists; sentence→evidence alignment is explicitly P3. The charspan data is extracted in `tree.py` — then dropped | Mark ◐, or populate charspans (the data is already there) |
| DESIGN §6.18 ✅ "structured logging + OpenTelemetry spans" | No OTel import/dep/span anywhere; stdlib `logging` with free-text messages; no run ids, no token counts | Drop the claim, or add an optional ~50-line OTel wrapper |
| DESIGN §6.14 ✅ caching "DocIndex, outline, and per-node summaries" | DocIndex LRU is real; the outline is re-rendered every selection call (cheap, but not cached); glosa never *generates* summaries (by design), so there is nothing to cache absent host enrichment | Reword |
| DESIGN §9 decisions: deps "docling-core, httpx, pydantic … in the core" | Contradicts §2 and pyproject: docling-core is test-only (correctly) | Fix the table |
| DESIGN P1 "87 tests", "StudioProjection" | 210 tests; the class is `DoclingProjection` | Sweep stale text |
| "mypy --strict clean" as an API property | True internally, but **no `py.typed` marker** — PEP 561 makes the package untyped for every consumer; Studio's own strict mypy would error on `import glosa` | Add `src/glosa/py.typed` + `Typing :: Typed` classifier |

---

## 8. Verification gaps

The suite is strong on what it covers (210 tests, ~90 % lines, behavior-named,
recovery paths exercised) and stops exactly where the headline claims start:

- **The `BudgetExhausted` machinery has zero coverage.** All three `raise`
  sites and the loop's `except` handler are unexecuted; every "ceiling" test
  exits via the pre-check path. No test sets a finite deadline
  (`deadline_s=None` everywhere). B15 (cheap-path crash) sat precisely in
  this shadow. Cancellation is claimed shipped and untested.
- **`INSUFFICIENT_EVIDENCE` is never produced by any test.** One of the four
  headline outcomes — and its legacy prefix rendering — could regress into
  the exact conflation the four-valued status exists to prevent, green.
- **The CLI is a shipped entry point at 0 % coverage.**
- **Contract tests cannot see Studio drift.** The scoping oracle is built
  from glosa's *own* `tree.py`; the tree-parity test pins behaviour "taken
  from docstrings"; no upstream commit SHA is recorded anywhere. By
  construction the suite goes red only when a glosa developer edits glosa —
  never when Studio moves. → Record the transcribed upstream revision per
  contract module; add a scheduled CI job that hashes the three mirrored
  Studio files and fails on change; best: capture Studio's real
  projection/graph payloads as golden fixtures (see §10).
- **Fixtures are toys.** All hand-built, 2–4 sections, 1–2 pages. Never
  exercised: list groups (the oracle's list-specific branch is dead), charts,
  `key_value_area`/`form_area`, page-less docs (markdown/HTML conversions —
  `_build_whole_document` via `_build` is uncovered), spanning tables, the
  >4 000-cell fallback, any document at realistic scale, **any committed
  output of a real conversion**.
- **Every reading-quality claim is measured against a scripted fake** whose
  own conformance to the `ChatModel` port is never checked
  (`assert_is_chat_model` is defined and never called; mypy does not check
  `tests/`). Real-model failure modes no scripted double can catch:
  per-backend constrained-decoding quirks, context overflow (nothing measures
  rendered prompt size), quote paraphrase rates — i.e. whether `grounded` is
  signal or noise.
- **No eval harness.** P2 is marked ✅ done while its own acceptance criteria
  ("20-question benchmark, ≥30 % fewer tokens, ≥40 % lower p95, ≥1 correct
  abstention") have never been run and cannot be (no corpus, no token
  accounting). No opt-in live-backend test exists, so DESIGN P1's "not yet
  verified" list has no executable path to ever flip.
- **No property-based tests** where they are cheapest (packing, folding,
  outline fitting, tree parity on generated trees). One pinned-slack example:
  `len(excerpt.text)` can exceed `char_budget` (separators and markers are
  not counted).

---

## 9. Strategic assessment

**Where the "no RAG" bet holds.** For *one document, point questions, a local
model, an auditing viewer*, the design is genuinely stronger than both
neighbours: no index to build or drift, provenance for free, abstention as an
outcome, and the reading loop bounded and inspectable. The long-document
mechanics (outline fitting, descent, packing) are the part of the bet that is
already won in code — modulo B2.

**Where it structurally loses.**
- *Aggregation/exhaustive queries* ("list every termination right"): a run
  reads ≤ `max_steps × excerpt_char_budget` ≈ 48 KB of the document and
  composes from ≤ 8 notes of ≤ 600 chars; `sufficient` is judged per unit, so
  one locally-sufficient read can end the run with a partial list as
  `ANSWERED`. Long-context stuffing or high-k embeddings win on recall here.
  Nothing in the docs names the query classes. → Name them; add an explicit
  coverage mode (map-reduce over matching units) later.
- *Corpus scale*: level 2 is deliberately unbuilt, and P4 plans to reuse
  Studio's OpenSearch vectors — embeddings return exactly where they win
  (routing, paraphrase recall). The honest position — "no index needed *per
  document*; indexes welcome when the host has one" — is in the design and
  absent from the README pitch.

**The moat and its cost.** The durable differentiator is projection parity +
the grounded trace *for a viewer* — and it is also the single-consumer
dependency: elem:: ids, `LABEL_MAP`, flat scoping are Studio-isms with no
value elsewhere, Studio has **never actually run glosa** (the E2E and the Vue
overlay are admitted-unverified), and the parity is pinned to a
hand-transcribed oracle of a frontend file glosa cannot see. Most other README
table rows (async, schema decoding, providers, public API) are commodity
engineering one upstream release could erase — and DESIGN §9.2's own plan to
upstream the bug fixes accelerates that erosion.

**The eval vacuum is the biggest strategic exposure.** PageIndex — the
best-known vectorless competitor — markets 98.7 % on FinanceBench. glosa's
central bet (free leads + BM25 beat paid per-node LLM summaries) is
unfalsified even internally. There is no number to cite, and the code cannot
currently produce one (no token accounting, no corpus).

**Grounding can be over-trusted.** `grounded=True` requires only a folded
substring match of a possibly two-word quote against a possibly truncated
excerpt; a true-but-irrelevant sentence gets a green check, and the L1 legacy
projection **drops the verdict entirely** — the flagship credibility signal
is invisible at the only shipped integration level. The README row states the
check without the caveat the design states.

---

## 10. Roadmap

### P0 — Correctness on the shipped promises (days)

1. **B1** scoping: explicit-parent exemption + discriminating fixture.
2. **B2** packing: truncate the top-ranked oversized element; honest marker.
3. **B4** error containment: compose/expansion/single-read failures degrade,
   never destroy the trace; map `BackendError`/`DocumentParseError` at the
   adapter boundary (factories), document the escape set in INTEGRATION.md.
4. **B5** make the deadline bind: clamp per-request timeout to
   `remaining_s`; split the CLI flag.
5. **B6** retry only retryable statuses; give `BackendError` fields
   (`status_code`, `retryable`).
6. **B3** typed errors on malformed JSON (visited set in `dfs_order`,
   defensive coercion, wrap `_build`).
7. **B7 + B10** French folding + quote folding (apostrophes, ligatures) —
   small diffs, large effect on the target corpus.
8. The one-line fixes: B11 (`gap_notes=0`), B13 (honest hedge reason), B14
   (expansion-attempted flag), B15 (cheap-path try), B16 (`has_sections` on
   readable elements), B19 (exit codes), `py.typed`.

### P1 — Credibility (1–2 weeks)

9. **Eval harness + corpus** (gates every claim): ~5 real conversions
   (FinanceBench PDFs convert cleanly and give an externally comparable
   number), 20 questions incl. deliberately unanswerable ones; measure
   hit-rate@k, abstention accuracy, grounded-quote precision, calls, tokens
   (requires wiring `usage`/`eval_count` through — do it here), p50/p95 vs
   `tests/baseline.py` and vs naive full-context stuffing. Publish the
   numbers in the README.
10. **Real Studio E2E once**: run a trace through the actual overlay; capture
    Studio's real projection/graph payloads for 2–3 documents as golden
    contract fixtures (parity by capture, not transcription); record upstream
    SHAs in the contract modules; scheduled CI hash-check on the three
    mirrored Studio files.
11. **Docs truth pass** (§7 table): the four overstated ✅ claims, the
    INTEGRATION.md wire-up, provider matrix wording, README streaming row,
    stale counts/names.
12. **Publish 0.1.0 to PyPI**; `uv build` in CI; version single-sourced.
13. **Gate calibration** (with the eval to measure it): stopword set,
    magnitude-aware margin, minimum-evidence floor for single-hit shortlists,
    original-views-only confidence. Surface `grounded=False` in the legacy
    `response` (marker line, like `STATUS_PREFIX`) so L1 users see the
    flagship signal.

### P2 — Product depth (after P1, ordered by the eval)

14. **Sentence→span alignment** (P3's evidence spans): the scaffolding is
    ready (per-element `ExcerptPart`s, verified quotes, charspans extracted
    and currently dropped). Upgrades `grounded` from a boolean to a
    highlightable rectangle — the demo the positioning depends on, and it
    makes §6.8's claim true.
15. **Soften the visited ban**: one bounded re-read on changed focus, setting
    the already-existing `revisited` flag.
16. **Streaming/typed events** (P3) — only after the loop's error containment
    (P0.3), which it depends on.
17. **Aggregation mode**: named query classes + a map-reduce coverage mode
    with per-section attribution; the natural stepping stone to level 2's
    route → read → reconcile.
18. Robustness backlog: table fallback (B9), page-mode orphans and titles,
    excerpt-cache bound, `health()` twin lifecycle, OpenAI payload knobs
    (temperature/max_tokens portability), property-based tests for
    packing/folding/outline, CLI tests, fixture realism (lists, charts,
    page-less, spanning tables, one large document).

---

*Full finding details (61 findings: severity, kind, evidence, suggested fix
per finding) are preserved in the audit working notes; every **[confirmed]**
item above was reproduced against `f358511` before being recorded here.*
