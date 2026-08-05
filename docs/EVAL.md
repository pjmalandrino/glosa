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

The bench is the number. **Forty papers, five questions each — 200 items.**
Light in the sense that matters: it runs in about two hours on a laptop with a
local 8B, and its scoring half runs in seconds with no GPU at all.

Scientific articles, and not by default. A paper is the document shape that
breaks readers in the most interesting ways at once — a hierarchy that goes
three levels deep, results that live in table cells rather than sentences,
definitions in the method used twenty pages later in the discussion, and half
the specifics banished to an appendix. It also brings one problem no contract
corpus has, and §3.5 is mostly about that problem: the model has probably
already read the paper.

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
# corpus/items/2603-00417-scaling-laws-for-sparse-retrieval.yaml
- id: 2603-00417-q02
  doc: 2603-00417-scaling-laws-for-sparse-retrieval
  kind: table                  # lookup | table | crossref | peripheral | not_stated
  question: "What recall@100 does the 340M model reach on the out-of-domain split?"
  options:
    A: "61.4"
    B: "68.9"
    C: "72.3"
    D: "55.0"
  answer: B
  gold_refs: ["#/tables/2"]    # where the answer lives, in Studio's projection
  gold_quote: "340M"
  distractor_refs:             # where the wrong values were taken from
    A: "#/tables/2"            # another row — on a table item, that is the point
    C: "#/tables/1"
    D: "#/texts/188"
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

### The five kinds, one per paper

Every paper owes exactly one item of each kind. That quota is checked by the
linter, and it is the single rule that keeps a 200-item suite from decaying:
forty papers authored over a week drift toward whatever is quickest to write,
which is `lookup`, and a suite of lookups is a keyword-matching test wearing a
benchmark's clothes.

| kind | where the answer is | what it separates |
| --- | --- | --- |
| `lookup` | one section of running prose | the baseline everything else is read against |
| `table` | a cell of a results table | glosa serializes tables as HTML, `docling-agent` flattens them outside page mode, PageIndex sees whatever the markdown export produced. Right table, wrong row scores zero |
| `crossref` | two sections — a symbol defined in the method, used in the results | one hop versus two, and whether a reader can carry a definition forward |
| `peripheral` | a figure caption, a footnote, an appendix | a reader that walks headings from the top and stops when it has enough never arrives. On a paper this is the most common real failure |
| `not_stated` | nowhere — `E` is correct | hallucination, and whether a typed abstention exists at all |

Five kinds x forty papers is 200 items with a per-kind breakdown of forty each
(§5), which is where the headline stops being one number and starts being an
explanation.

---

## 3. The fairness contract

The comparison is worthless if the three engines differ in anything but their
reading loop. Five confounds, and what neutralises each:

### 3.1 One parse

All three read **the same `DoclingDocument`**, converted once from the PDF the
manifest pins. They just need it in different shapes:

| Engine | Input | Produced by |
| --- | --- | --- |
| `glosa` | `docling.json` (serialized `DoclingDocument`) | the corpus, verbatim |
| `docling-agent` | a `DoclingDocument` object | `DoclingDocument.model_validate_json` of the same file |
| PageIndex | `doc.md` via `--md_path` | `export_to_markdown()` of the same file |

PageIndex's own PDF parser is therefore **not** in the comparison. That is
deliberate: it is a good parser, and on two-column papers with equations the
difference between parsers is enormous — which is exactly why letting it run
would mean measuring Docling against PyMuPDF and reporting it as a reasoning
result. `gbench fetch` keeps `source.pdf` next to the conversion so a future run
*can* measure that, as a separate, honestly-labelled experiment (§8, open 2).

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
the bench into a test of common sense at a 20 % floor. The linter rejects them
(§6).

One kind inverts the rule, and the linter knows it: on a `table` item the
distractors are **other rows of the same table**, and that is the entire
difficulty — find the right table, then the right row. Forbidding the overlap
there would forbid the only good distractors the kind has.

### 3.5 The controls

Five numbers that surround the accuracy figure and stop it lying. Three of them
are "engines" implementing the same port, with no navigation at all:

| Control | What runs | What it answers |
| --- | --- | --- |
| **closed-book** | the model, the item, **no document** | how much of the score is the model's priors? |
| **abstract-only** | the model, the item, **the title and abstract** | how much is answerable from the paper's own summary, with no navigation at all? |
| **oracle-context** | the model, the item, **only `gold_refs`' text** | the ceiling this model can reach on this item. `oracle − engine` is *retrieval headroom*; `100 − oracle` is the model's own limit, which no reading loop can fix. |
| **rotation** | the substantive options cyclically rotated A→B→C→D | position bias, two ways — see below. |
| **not-stated** | 40 items where the value is absent from the paper | hallucination. The correct answer is `E`. |

An item either control answers is marked `leaky` and drops out of the headline,
reported on its own line.

**On a paper corpus, closed-book is the whole ballgame.** The model has very
likely seen the paper, or its abstract, or a thread about it. A benchmark built
on arXiv without this number is measuring memorisation and calling it retrieval,
and the fix is in the corpus rather than the metric: `gbench manifest` takes a
date window, and the window should start after the model's training cutoff
(§6). The control is what confirms that worked, per item.

**abstract-only exists because papers come with a summary.** Every one of them
opens with 200 words that answer the obvious questions, so an item answerable
from the abstract makes navigation free — read the first section, stop, score.
The linter refuses items whose `gold_refs` are in the abstract; the control
catches the ones answerable from it anyway, by paraphrase, which no ref check
can see.

`E` does not rotate. It is a sentinel, not a candidate: moving it would change
what "abstain" costs from item to item, and the abstention mapping below has to
name one letter.

**Rotation, at 200 items.** Sweeping all four rotations of every item is 800
executions per engine and six engines to run, and most of it buys nothing. What
sweeping everything defends against is *aggregate* position bias — the suite
systematically putting right answers in the same slot — and a suite of 200 gets
that for free by giving each item a deterministic assigned rotation
(`sha256(item_id) % 4`, uniform across the suite, stable across runs). What it
does not get for free is *per-item* stability: whether an engine changes its
mind when the options move. So 40 items, stratified across kinds and fixed
between runs, are swept in full.

320 executions per engine instead of 800, and both numbers rotation exists to
produce still come out. `flip_rate` is reported with the `n` it was computed
over, and is `—` rather than 0 % when nothing was swept: an unswept item would
otherwise report as "did not flip", and a benchmark would publish perfect
stability having looked at nothing.

The `not_stated` items are where abstention stops being a design opinion and
becomes a number. Each engine gets its best available abstention signal,
declared in the adapter rather than hidden in the scorer:

| Engine | Signal mapped to `E` |
| --- | --- |
| `glosa` | `status ∈ {not_in_document, insufficient_evidence}` |
| `docling-agent` | `converged == False` |
| PageIndex | none — it has no such signal |

`budget_exhausted` is deliberately not in glosa's row: running out of budget is
a failure to read, not a finding about the document, and mapping it to `E` would
pay an engine for giving up.

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

**The corpus is a manifest, not a pile of PDFs.** Forty papers and their
conversions are hundreds of megabytes, which does not belong in git. What is
committed is `manifest.yaml` (arXiv id, version, **SHA-256**, licence,
category), the items, and a `sections.json` per paper — ref to text of the
projection, around 100 KB, and exactly what the linter checks quotes against.
The PDFs and conversions are rebuilt by `gbench fetch` + `gbench convert`.

Three things fall out of that, and all three are the point. A third party can
verify every quote, every distractor and every published number with nothing
downloaded and no GPU. The SHA-256 means a run from six months ago and a run
today either read the same bytes or fail loudly — arXiv versions do get
replaced under their own id. And `sections.json` can drift from the conversion
it was generated from, so a test fails if it ever does.

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

Two hundred items across six engines is 1 920 executions even under the
sampling plan of §3.5; served serially by a local 8B that is most of a day. The harness runs items concurrently — and that
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

Per engine, with the controls beside it, and then the same accuracy broken down
per kind and per domain:

**Answer quality**
- `accuracy` — mean over items of the mean over that item's rotations, and
  `lift = accuracy − closed_book`
- `headroom = oracle_context − accuracy` — how much of the gap is navigation
- `flip_rate`, with the `n` it was computed over — `—` when nothing was swept
- `unparsed_rate` — no letter recoverable from the answer text; counts as wrong
  *and* is reported alone, because it is the `find_json_dicts[0]` failure class
  (`DESIGN.md` §4.1) expressed as a percentage
- on `not_stated` items: `abstention_recall` and, on answerable items,
  `false_abstention` — reported as a pair, because either alone is gameable by
  an engine that always abstains

**Navigation** — free, because every item carries `gold_refs`
- `hit@k` — did the engine read a gold ref at all, within its first k reads?
- `read_precision` — gold chars ÷ chars read. The cost of over-reading.
- `answer_without_hit` — right letter, never read the gold section. Guessing
  that lands, and a number worth watching on a 20 %-floor format.

**Cost**
- `llm_calls`, `prompt_chars` — cold (with `prepare`) and warm (without)
- `p50/p95 wall_s` — from the serial latency pass only

**Engine-specific**, rendered as `—` for engines that cannot report it
- `grounded_rate`, `false_citation_rate` — `glosa`'s `Step.grounded`
- `steps`, `revisits`, `fallback_rate`

**Breakdowns**, MMLU-style — engines down the side, kinds and arXiv categories
across the top. This is where a headline stops being one number: an engine 20
points behind on `table` and level everywhere else has a serializer problem, not
a navigation problem, and the average cannot tell you which.

### The statistics of 200 items

Two hundred is enough to say something and not enough to say everything, and
the report is built to keep those apart.

**The headline.** At 200 items an accuracy near 70 % carries roughly a
±6.4-point 95 % interval — tight enough to separate engines that differ by ten
points, not by four. The real power comes from **pairing**: every engine answers
the *same* items, so the comparison is `A right / B wrong` versus
`A wrong / B right`, which has far more power than two independent proportions.
The harness reports paired deltas with a bootstrap interval over items (20
lines, no `scipy`), and prints `Δ = +6 pts [−1, +13] — not separable` in exactly
that form. A benchmark that cannot say "these two are tied" is one that will
eventually be used to claim they are not.

**The breakdowns are weaker, and are printed with their `n`.** Forty items per
kind is about ±14 points; twenty-five per domain is about ±18. Those columns
are for finding a hypothesis, not for settling one — a per-kind gap worth
believing is a large one, and the honest follow-up to a suggestive column is
more items of that kind, not a stronger claim about the one you have.

**An item is one observation, not four.** The 40 swept items are measured four
times each; accuracy is the mean over items of the mean over rotations, and
every interval resamples items. Treating 320 executions as 320 observations
would report an interval about 40 % too narrow.

---

## 6. The corpus: forty papers

**Selection is a command, not a taste.** `gbench manifest` harvests arXiv's
OAI-PMH interface — the one endpoint that reports a per-record licence — over a
date window and a list of sets, and keeps the first five redistributable papers
per set. Eight sets gives forty papers and eight columns in the per-domain
breakdown.

```bash
uv run gbench manifest --sets cs.CL,cs.LG,math.ST,q-bio.NC,physics:cond-mat,eess.SP,stat.ME,econ.EM \
                       --since 2026-03-01 --until 2026-05-31 --per-set 5
uv run gbench fetch      # downloads the PDFs and pins their SHA-256
uv run gbench convert    # Docling → docling.json, doc.md, sections.json
```

Three selection rules, each of which rules out papers we would otherwise want:

1. **CC-BY, CC-BY-SA or CC0 only.** arXiv's default licence lets *arXiv*
   redistribute the paper, not us — and a number computed over a paper a third
   party cannot read is a number nobody can check. This rejects most of any
   harvest, which is why `manifest` prints what it dropped. `gbench lint`
   re-checks it, as an error.
2. **Posted after the model's training cutoff.** Otherwise the closed-book
   control is measuring how famous the paper is. The window is an argument
   precisely so it moves when the model does.
3. **Long enough to need navigating.** A four-page workshop note is answerable
   by pasting the whole thing into the context, which measures nothing —
   glosa's own `direct_char_threshold` would skip its reading loop entirely.

**Five questions per paper, one of each kind** (§2). Two hundred items, forty
per kind, twenty-five per domain.

**We are writing the benchmark our own engine is measured on.** That is a real
conflict of interest, and the only honest response is to make cheating
mechanically visible. `gbench lint` fails the suite on:

1. a `gold_quote` that is not literally in the text of its `gold_refs`;
2. a distractor that **claims** a source ref and is not in it — an unverifiable
   provenance note is worse than none, because it looks like evidence;
3. a distractor invented rather than taken from the paper (on the kinds whose
   options are values), or one sitting inside a `gold_ref` (on the kinds where
   that makes two answers defensible — not `table`, see §3.4);
4. an item whose `gold_refs` are in the **abstract** — navigation would be free;
5. two items of the same kind on one paper, or a paper missing a kind;
6. an unbalanced answer key — χ² over the letter distribution;
7. the longest-option artefact — the correct answer being the longest string
   more often than chance;
8. a paper whose licence does not permit redistributing its text;
9. an item answerable closed-book or from the abstract, once those controls have
   run: flagged `leaky`, excluded from the headline, reported on its own line;
10. an item authored from a trace rather than from the paper — `authored_from`
    is a required field and `trace` is not an accepted value.

Plus the two things that keep it re-checkable by someone who does not trust us:
the manifest, the items, the `sections.json` and the journals are committed, so
any result in this repository can be re-scored — and re-verified against the
paper's own text — without a GPU and without downloading anything; and the
`not_stated` items name the ref the value *would* have been in, so the claim
"it is absent" is auditable rather than asserted.

---

## 7. Phases

| | | |
| --- | --- | --- |
| **B0** | domain + controls ✅ | items, the five kinds, rotation and the sampling plan, the letter parser, the metrics, the corpus linter, the arXiv selection — all pure and tested. `glosa` plus the three controls. Journal, `score`, breakdowns. Runs end to end on a generated fixture with a scripted model, in CI, no GPU and no network. |
| **B1** | the corpus | `gbench manifest` over eight sets, `fetch`, `convert`, then 200 authored items with the linter green. The gate is authoring discipline and about a day of reading papers, not code. |
| **B2** | the competitors | `docling-agent` and PageIndex adapters behind their extras, pinned by contract tests against an installed version; `prepare()` cost measured; first three-way table. |
| **B3** | in the loop | nightly on a self-hosted box; `bench/RESULTS.md` regenerated; P2's acceptance criteria (`DESIGN.md` §7 — "≥30 % fewer tokens, ≥40 % lower p95, ≥1 correct abstention") finally have a number. |

B0 has no external dependency and no LLM: the scripted-model path means the
whole harness is exercised by `pytest` on every push. That ordering is
deliberate — the parts that can silently be wrong (a scorer, a metric, a letter
parser, a rotation) are the parts that get tests, and the parts that cannot be
tested (does an 8B read a paper) are the parts that get a number.

**Cost of one full run**, for planning. 320 executions per engine (§3.5), six
engines:

| | executions | LLM calls |
| --- | --- | --- |
| three controls | 960 | ≈ 960 (one each) |
| `glosa` | 320 | ≈ 640 |
| `docling-agent` | 320 | ≈ 1 900 (two per section read) |
| PageIndex — queries | 320 | ≈ 1 300 |
| PageIndex — **index** | — | ≈ 3 200 one-off (a call per node, 40 papers) |

≈ 8 000 calls; at 3 s each and concurrency 4, a little under two hours. Note
which line dominates: PageIndex's index costs more than every question every
engine asks, put together. That is the trade its design makes, and `prepare()`
exists so the table shows it instead of hiding it in a latency average.

`--profile smoke` (12 items, no sweep) is about three minutes and is what runs
before a commit.

---

## 8. Decisions

| Question | Decision |
| --- | --- |
| Corpus | **40 arXiv papers, CC-BY or better, posted after the model's cutoff.** Selected by a command, not by taste. |
| Size | **5 questions per paper, one of each kind = 200 items.** Enough to separate engines by ten points; the report refuses to claim four. |
| Scoring | **MCQ, exact match.** No LLM judge — see §1. |
| Where it lives | **`bench/`, its own project and lockfile.** Competitor dependencies never touch `glosa`'s. |
| What is committed | **The manifest, the items, `sections.json`, the journals.** Not the PDFs or the conversions — `fetch` + `convert` rebuild them from pinned bytes. |
| Rotation | **Assigned per item, swept on 40.** Aggregate position bias for free, per-item stability on a sample. |
| Answer extraction | **One shared deterministic parser**, no repair prompt. An engine that cannot emit a letter scores zero and gets a named metric for it. |
| Abstention | **Per-engine declared mapping**, plus a strict-letters re-score of the same journal. |
| Wall-clock | **Serial pass only.** Contended timings are recorded and refused for ranking. |

Open:

1. **Which eight arXiv sets.** The per-domain breakdown is only interesting if
   the fields differ in how they write — a maths paper puts everything in
   numbered theorems, an experimental physics paper in figure captions, an
   econometrics paper in tables. The shortlist in §6 is a first guess and should
   be revisited once there is one real run to look at.
2. **Whether the fixture stays.** `toy-ccap` is a generated contract, not a
   paper, and it exists so CI has something to run. Once B1 lands, one small
   CC-BY paper could replace it and make the CI path identical to the real one
   — at the cost of committing a `sections.json` heavy enough to notice.
3. **Whether PageIndex's own parser gets its own run.** §3.1 excludes it to
   isolate reasoning, and on two-column papers with equations that exclusion is
   doing a lot of work. A second, separately-labelled configuration would
   measure the pipeline end to end, which is what a Studio user actually buys.
   Cheap to add once B2 lands; must never be merged into the same table.
