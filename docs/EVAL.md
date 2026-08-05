# The bench — an MMLU-light for document readers

> Companion to [`DESIGN.md`](DESIGN.md). Closes §6.19 ("eval harness — every
> phase has a number attached") and open question §9.1 ("benchmark corpus").
> Diagram: [`architecture.drawio`](architecture.drawio) page 5.

Three engines claim to answer a question over one document:
[`docling-agent`](https://github.com/docling-project/docling-agent)'s
`DoclingRAGAgent`, [PageIndex](https://github.com/VectifyAI/PageIndex)'s
tree search, and `glosa`. They disagree about what a document map is, about
where the LLM calls should go, and about whether abstention exists. Nothing in
this repository currently says which of those disagreements matter.

The bench is the number. It is deliberately **light**: one afternoon to author,
under an hour to run on a laptop with a local 8B, and cheap enough to re-run on
every change to the reading loop.

---

## 1. What it measures — and what it refuses to measure

**Measured.** Given the same document and the same model, does the engine
*navigate to the passage that holds the answer*, and does it come back with the
right value? Plus what that cost: LLM calls, characters of prompt, wall-clock.

**Not measured.** Prose quality. Summarisation. Writing, editing, enriching —
`docling-agent` does those and `glosa` deliberately does not (`DESIGN.md` §1),
so a benchmark covering them would compare a tool to a non-tool.

**The choice that follows from that: multiple choice, scored by exact match.**

The obvious alternative is free-text answers scored by an LLM judge. Rejected,
for three reasons that are worth writing down because the trade is real:

| | MCQ + exact match | free text + LLM judge |
| --- | --- | --- |
| Cost of scoring | zero | ≈ one more run |
| Reproducible in six months | yes, it is `==` | no, the judge drifts |
| Fourth model in the comparison | no | yes, and it has its own biases |
| Measures generation quality | **no** | partially |

The third row is the decisive one. We are comparing *navigation*, and a judge
would introduce a model nobody controls into the very number meant to settle a
disagreement between three engines. The fourth row is the price paid: MCQ
measures *discrimination*, not *generation*. An engine that finds the right
paragraph and writes an awkward sentence about it scores full marks here. That
is acceptable, because the awkward sentence is a model property and the right
paragraph is an engine property.

MMLU's shape, then, minus its subject matter: **the answer must come out of
*this* document, not out of the model's weights.** Everything in §3 exists to
enforce that one sentence.

---

## 2. One item

```yaml
# corpus/items/marche-public-2019.yaml
- id: mp2019-q03
  doc: marche-public-2019
  kind: lookup                 # lookup | crossref | table | not_stated
  question: "Quel est le taux des pénalités de retard prévu au contrat ?"
  options:
    A: "1/3000 du montant du marché par jour de retard"
    B: "1/1000 du montant du marché par jour de retard"
    C: "5 % du montant du marché, forfaitaires"
    D: "1/1000 du montant du lot concerné par jour de retard"
  answer: B
  gold_refs: ["#/texts/94"]    # where the answer lives, in Studio's projection
  gold_quote: "Les pénalités de retard sont fixées à 1/1000 du montant"
  distractor_refs:             # where the wrong values were taken from
    D: "#/texts/131"
  authored_by: pjm
  authored_from: document      # never: from a trace
```

Four substantive options, plus a fifth on **every** item:
`E: "Not stated in this document"`. It is the correct answer on the `not_stated`
items (§3.5) and wrong everywhere else.

Present on every item, not only where it is right — otherwise its mere presence
announces the answer, and the bench would be measuring whether an engine
noticed that the option list grew. It also means the guessing floor is 20 %, and
that an engine can lose points by over-abstaining, which is the correct
incentive and is measured as `false_abstention` (§5).

The question handed to every engine is the *rendered* item: the question text,
then the options, then a single instruction to answer with a letter. Byte for
byte the same string for all three engines. That is the whole prompt-level
fairness story, and it is the only prompt the bench controls — everything an
engine does after receiving it *is the thing under test*.

`gold_refs` is what makes this more than an accuracy number: it is a free
retrieval label on every item, and it is what §5's navigation metrics are
computed from.

---

## 3. The fairness contract

The comparison is worthless if the three engines differ in anything but their
reading loop. Five confounds, and what neutralises each:

### 3.1 One parse

All three read **the same `DoclingDocument`**, converted once and committed.
They just need it in different shapes:

| Engine | Input | Produced by |
| --- | --- | --- |
| `glosa` | `docling.json` (serialized `DoclingDocument`) | the corpus, verbatim |
| `docling-agent` | a `DoclingDocument` object | `DoclingDocument.model_validate_json` of the same file |
| PageIndex | `doc.md` via `--md_path` | `export_to_markdown()` of the same file |

PageIndex's own PDF parser is therefore **not** in the comparison. That is
deliberate: it is a good parser, but if it ran we would be measuring Docling
against PyMuPDF and reporting it as a reasoning result. The corpus keeps
`source.pdf` next to `docling.json` so a future run *can* measure that, as a
separate, honestly-labelled experiment.

### 3.2 One model, one endpoint

An Ollama endpoint, one model id, `temperature=0`, fixed seed. All three reach
it:

- `glosa` — `OllamaChatModel` natively;
- `docling-agent` — `create_backend(BackendConfig(type="ollama", base_url=…))`;
- PageIndex — LiteLLM, model string `ollama/<model>`.

Schema-constrained decoding stays **on** where an engine has it and off where it
does not. It is a property of the engine, not of the setup — `glosa` shipping
constrained decoding and `docling-agent` shipping a regex over a ` ```json `
block is precisely the kind of difference the bench should show, as a parse
failure rate (§5).

### 3.3 One budget

Same per-item ceiling for everyone: max LLM calls, max wall-clock. An engine
that blows it does not get a longer leash; the item is scored as `budget` — an
outcome, not an error. `glosa` enforces this natively (`Budget`),
`docling-agent` through `max_iterations`, PageIndex through the harness's own
deadline. Differences in how gracefully the ceiling is hit are visible in the
outcome mix.

### 3.4 Distractors from the same document

Every wrong option is a **real value taken from another section of the same
document** (`distractor_refs` records where). This is the single most important
authoring rule, and it exists to make the bench able to hurt the home team:

- a keyword-overlap prior — `glosa`'s BM25 shortlist — cannot win by lexical
  luck, because the distractor's vocabulary is in the document too;
- the model's parametric priors cannot win by plausibility, because all four
  options are plausible *for this document*;
- an engine that reads the right *section* but the wrong *row of the table*
  now scores zero, where a free-text judge would have given it partial credit.

Options invented by an author — round numbers, obviously silly amounts — turn
the bench into a test of common sense at 25 % floor. The linter rejects them
(§6).

### 3.5 The controls

Four numbers that surround the accuracy figure and stop it lying. Two of them
are "engines" implementing the same port, with no navigation at all:

| Control | What runs | What it answers |
| --- | --- | --- |
| **closed-book** | the model, the item, **no document** | how much of the score is the model's priors? An item answered without the document is marked `leaky` and reported separately. |
| **oracle-context** | the model, the item, **only `gold_refs`' text** | the ceiling this model can reach on this item. `oracle − engine` is *retrieval headroom*; `100 − oracle` is the model's own limit, which no reading loop can fix. |
| **rotation** | each item run with its four substantive options cyclically rotated A→B→C→D | position bias. Score is the mean over rotations; the spread is reported as `flip_rate` — an engine that changes answer when the options move is not reading, it is guessing. |
| **not-stated** | ~20 % of items where the value was removed from the document | hallucination. The correct answer is `E`. |

`E` does not rotate. It is a sentinel, not a candidate: moving it would change
what "abstain" costs from item to item, and the abstention mapping below has to
name one letter.

The `not_stated` items are where abstention stops being a design opinion and
becomes a number. Each engine gets its best available abstention signal,
declared in the adapter rather than hidden in the scorer:

| Engine | Signal mapped to `E` |
| --- | --- |
| `glosa` | `status ∈ {not_in_document, insufficient_evidence}` |
| `docling-agent` | `converged == False` |
| PageIndex | none — it has no such signal |

That last row is a finding, not a handicap; the bench reports the mapping next
to the score so nobody has to guess. And because scoring runs offline over the
journal (§4.3), the same run is *also* scored under a strict policy where only
a literal `E` in the answer text counts. Two numbers, one execution, and the
gap between them is exactly the value of a typed status.

---

## 4. Architecture

### 4.1 It is a separate project, and that is load-bearing

```
glosa/                     httpx + pydantic. Unchanged. Untouched.
bench/                     its own pyproject.toml, its own uv.lock
  └── depends on glosa by path, and on nothing else by default
      extras:  [docling-agent] → mellea, docling-agent
               [pageindex]     → litellm, pymupdf, PyPDF2, pyyaml
```

`DESIGN.md` §9 says "`docling-core`, `httpx`, `pydantic`. Nothing else, ever, in
the core". A benchmark that pulls `mellea` and `torch` into the root lockfile to
measure a competitor would break that rule to prove a point about it. So the
bench is a sibling project with a path dependency, competitor adapters live
behind extras, and every adapter imports its engine **inside** the constructor —
`uv run gbench run --engines glosa,closed-book` works with nothing installed but
`glosa`.

### 4.2 Layers

Same hexagon as the package it measures — `domain` pure, `ports` protocols,
`infra` adapters, and the CLI as composition root:

```
bench/src/gbench/
  domain/
    item.py        Item, Option, ItemKind, RenderedItem — and the rotation
    scoring.py     answer text → letter, letter + policy → Outcome
    metrics.py     accuracy, hit@k, read precision, percentiles, paired bootstrap
    lint.py        the corpus linter (§6) — pure predicates over a Suite
    report.py      capability-aware rendering: "—" is not 0
  ports/
    engine.py      Engine: name · capabilities · prepare(doc) · answer(item)
    corpus.py      Corpus: documents, their variants, the items
  infra/
    engines/
      glosa.py           in-process, no extra dependency
      docling_agent.py   lazy import, [docling-agent]
      pageindex.py       lazy import + subprocess for the index build, [pageindex]
      controls.py        closed-book and oracle-context, same port
    cache.py       content-addressed by (doc_hash, engine, config_hash)
    journal.py     append-only JSONL, one row per (item, engine, rotation)
  cli.py           gbench lint | run | score | report
corpus/
  suite.yaml
  docs/<slug>/{source.pdf, docling.json, doc.md, meta.yaml}
  items/<slug>.yaml
```

### 4.3 The two decisions inside that layout

**`prepare()` is separate from `answer()`.** PageIndex pays an LLM call per node
*before the first question*, to write the summaries its tree search navigates on;
`glosa` pays zero because it reads the host's projection; `docling-agent` pays
zero because it navigates headings. Folding that into per-item latency would
either slander PageIndex (all of it charged to item #1) or hide its cost
entirely (amortised into invisibility). So the port has two methods, the harness
times them separately, and the report shows **cold** cost (index + queries) and
**warm** cost (queries only) as two columns. Both are true; which one matters
depends on whether the document is read once or a hundred times.

**Running and scoring are different commands.** `run` executes engines and
appends raw rows to a JSONL journal: the full answer text, the trace summary,
timings, the config hash. `score` reads the journal and computes numbers. The
models are never invoked by `score`. Changing a scoring policy, fixing a letter
parser, or adding a metric is therefore free, and no result in this repository
is ever a number nobody can recompute. It also lands `DESIGN.md` §6.17
(deterministic replay) for the bench's own runs.

### 4.4 The `Engine` port

```python
@runtime_checkable
class Engine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities:
        """What this engine can report — read_refs, abstention, llm_calls,
        groundedness. A metric an engine cannot report is `None` in the
        journal and `—` in the report, never `0`."""

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        """Everything done before the first question. Cached by document hash."""

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        """One question. Never sees `item.answer`."""
```

`EngineAnswer` is the whole comparable surface: `text`, `read_refs`,
`abstained`, `llm_calls`, `prompt_chars`, `wall_s`, `error`, plus an opaque
`trace` blob kept for forensics. Three engines with three different trace
shapes flatten to this and nothing else, which is what makes the report a table
instead of three paragraphs.

### 4.5 Concurrency and the one thing it breaks

Sixty items × four rotations × five engines is 1 200 runs; served serially by a
local 8B that is hours. The harness runs items concurrently — and that
**invalidates wall-clock as a comparable metric**, because the engines are then
queueing behind each other on one GPU.

Stated rather than papered over:

- primary cost metrics are **`llm_calls`** and **`prompt_chars`**, which are
  invariant under contention;
- wall-clock rows produced under `--concurrency > 1` are flagged
  `contended: true` and the report refuses to rank on them;
- `gbench run --latency-pass` re-runs a fixed 12-item subset strictly serially
  for the p50/p95 numbers.

An engine whose whole claim is "≈2 round-trips instead of ≈10 sequential"
(`DESIGN.md` §6.12) must not be allowed to prove it with a contended stopwatch.

---

## 5. What comes out

Per engine, per item kind, with the controls beside it:

**Answer quality**
- `accuracy` — mean over rotations, and `lift = accuracy − closed_book`
- `headroom = oracle_context − accuracy` — how much of the gap is navigation
- `flip_rate` — share of items whose answer changes under option rotation
- `unparsed_rate` — no letter recoverable from the answer text; counts as wrong
  *and* is reported alone, because it is the `find_json_dicts[0]` failure class
  (`DESIGN.md` §4.1) expressed as a percentage
- on `not_stated` items: `abstention_recall` and, on answerable items,
  `false_abstention`

**Navigation** — free, because every item carries `gold_refs`
- `hit@k` — did the engine read a gold ref at all, within its first k reads?
- `read_precision` — gold chars ÷ chars read. The cost of over-reading.
- `answer_without_hit` — right letter, never read the gold section. Guessing
  that lands, and a number worth watching on a 25 %-floor format.

**Cost**
- `llm_calls`, `prompt_chars` — cold (with `prepare`) and warm (without)
- `p50/p95 wall_s` — from the serial latency pass only

**Engine-specific**, rendered as `—` for engines that cannot report it
- `grounded_rate`, `false_citation_rate` — `glosa`'s `Step.grounded`
- `steps`, `revisits`, `fallback_rate`

### The statistics of a light suite

Sixty items is small, and pretending otherwise would undo the point of the
exercise. On 60 items an accuracy near 70 % carries a ±11-point 95 % interval —
wide enough that two engines differing by 8 points are indistinguishable.

The fix is not more items, it is **pairing**. Every engine answers the *same*
items, so the comparison is `A right / B wrong` versus `A wrong / B right`, and
that has far more power than two independent proportions. The harness reports
paired deltas with a bootstrap interval over items (20 lines, no `scipy`), and
`report` prints `Δ = +6 pts [−1, +13] — not separable` in exactly that form.
A benchmark that cannot say "these two are tied" is a benchmark that will
eventually be used to claim they are not.

---

## 6. The corpus, and the linter that makes it trustworthy

Six documents × ten items = sixty. Documents chosen for the shapes that break
readers: a contract with numbered articles and no descriptive headings (kills
heading-only navigation), a technical report with tables, a scanned-then-OCR'd
PDF, a document with no headings at all, one non-English, one long (>100
sections) to put the outline over budget.

Item kinds: `lookup` (one section), `crossref` (needs two — a definition in one
article applied in another), `table` (a cell, not a paragraph), `not_stated`.

**We are writing the benchmark our own engine is measured on.** That is a real
conflict of interest and the only honest response is to make cheating
mechanically visible. `gbench lint` fails the suite on:

1. a `gold_quote` that is not literally in the text of its `gold_refs`;
2. a distractor that appears nowhere in the document (invented) or inside a
   `gold_ref` (ambiguous);
3. an unbalanced answer key — χ² over the letter distribution;
4. the longest-option artefact — the correct answer being the longest string
   more often than chance;
5. an item answerable closed-book, once the closed-book control has run:
   flagged `leaky`, excluded from the headline, reported as its own line;
6. items authored from a trace rather than from the document — `authored_from`
   is a required field and `trace` is not an accepted value.

Plus the two things that keep it re-checkable by someone who does not trust us:
the corpus and the journals are committed, so any result in this repository can
be re-scored by a third party without a GPU; and the `not_stated` items name
the ref the value *was* removed from, so the removal is auditable.

---

## 7. Phases

| | | |
| --- | --- | --- |
| **B0** | domain + controls | `Item`/rotation/scoring/metrics/lint as pure, tested code; `glosa`, `closed-book` and `oracle-context` engines; journal + `score`/`report`. Runs end to end on a 3-item toy suite with a scripted model, in CI, no GPU. |
| **B1** | the corpus | six documents, sixty items, linter green. The gate is authoring discipline, not code. |
| **B2** | the competitors | `docling-agent` and PageIndex adapters behind their extras; `prepare()` cost measured; first three-way table. |
| **B3** | in the loop | nightly run on a self-hosted box; `bench/RESULTS.md` regenerated; P2's acceptance criteria (`DESIGN.md` §7 — "≥30 % fewer tokens, ≥40 % lower p95, ≥1 correct abstention") finally have a number. |

B0 has no external dependency and no LLM: the scripted-model path means the
whole harness is exercised by `pytest` on every push. That ordering is
deliberate — the parts that can silently be wrong (a scorer, a metric, a
letter parser) are the parts that get tests, and the parts that cannot be
tested (does an 8B read a contract) are the parts that get a number.

**Cost of one full run**, for planning: ≈ 3 600 query-time LLM calls plus ≈ 500
one-off PageIndex indexing calls; about 45 minutes at concurrency 4 on a local
8B. `--profile smoke` (12 items, one rotation) is three minutes.

---

## 8. Decisions

| Question | Decision |
| --- | --- |
| Scoring | **MCQ, exact match.** No LLM judge — see §1. |
| Where it lives | **`bench/`, its own project and lockfile.** Competitor dependencies never touch `glosa`'s. |
| Corpus size | **60 items / 6 documents.** Light enough to re-run per change; too small to claim small differences, which §5 enforces rather than hides. |
| Answer extraction | **One shared deterministic parser**, no repair prompt. An engine that cannot emit a letter scores zero and gets a named metric for it. |
| Abstention | **Per-engine declared mapping**, plus a strict-letters re-score of the same journal. |
| Wall-clock | **Serial pass only.** Contended timings are recorded and refused for ranking. |

Open:

1. **The documents themselves.** They must be redistributable — a real
   engagement's PDFs cannot be committed. Public procurement contracts, EU
   regulations and published technical reports cover five of the six shapes;
   the OCR'd one may need to be produced from a public scan.
2. **Whether PageIndex's own parser gets its own run.** §3.1 excludes it to
   isolate reasoning. A second, separately-labelled configuration would measure
   the pipeline end to end, which is what a Studio user actually buys. Cheap to
   add once B2 lands; must never be merged into the same table.
