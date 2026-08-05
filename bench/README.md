# gbench — an MMLU-light for document readers

Compares [`docling-agent`](https://github.com/docling-project/docling-agent),
[PageIndex](https://github.com/VectifyAI/PageIndex) and
[`glosa`](..) on the same document, with the same model, under the same budget.

**Forty arXiv papers, five questions each — 200 items.** Multiple choice,
scored by exact match, surrounded by controls that stop the number lying: a
**closed-book** floor (the model has probably read the paper), an
**abstract-only** floor (papers come with a summary that answers the obvious
questions), an **oracle-context** ceiling, option **rotation** against position
bias, and distractors taken from the paper itself so keyword overlap cannot win.

Five kinds of question, one of each per paper — `lookup`, `table`, `crossref`,
`peripheral`, `not_stated` — so the headline stops being one number and starts
being an explanation.

Design and rationale: [`../docs/EVAL.md`](../docs/EVAL.md).

## Use

```bash
uv sync                              # glosa + pyyaml. Nothing else.
uv run gbench lint                   # keep the corpus honest — no model, no GPU
uv run gbench run                    # glosa + the three controls
uv run gbench score                  # declared-abstention policy
uv run gbench score --policy strict  # re-scores the same journal
```

Building or rebuilding the corpus needs the network, and `convert` needs
Docling. Papers are picked by hand — see
[`corpus/THEMES.md`](corpus/THEMES.md) for the eight themes and what to look
for — and pasted into a file, one id per line:

```bash
uv run gbench manifest --from-list corpus/papers.txt
uv run gbench fetch                  # downloads the PDFs, pins their SHA-256
uv sync --extra convert && uv run gbench convert
uv run gbench lint --verify-quotes
```

No licence hunting: under the default `fetch-only` policy the corpus commits
hashes rather than the papers, so any licence works.

Competitors live behind extras, and their adapters import inside the
constructor — so the two commands above work on a machine with neither
`mellea` nor `litellm` installed:

```bash
uv sync --extra docling-agent --extra pageindex
uv run gbench run --engines glosa,docling-agent,pageindex,closed-book,abstract-only,oracle-context
```

## Why it is a separate project

`glosa`'s runtime dependencies are `httpx` and `pydantic`, and that is a
promise the package keeps. Measuring a competitor must not put `mellea` or
`torch` in its lockfile to prove a point about not depending on them — so the
bench is a sibling project with its own `uv.lock` and a path dependency on
`glosa`.

## Layout

`domain/` is pure and tested — items, rotation, the letter parser, the metrics,
the corpus linter. `ports/` has `Engine` and `Corpus`. `infra/` holds the
adapters, the journal and the corpus reader. `runner.py` writes rows,
`scoreboard.py` reads them, and they never meet: **running and scoring are
different commands**, so any number here can be recomputed from a committed
journal without a GPU.

## What is committed

The manifest (arXiv id, version, **SHA-256**, licence, category), the items, the
journals, and a `refs.json` per paper — ref, label, character count and a hash,
no text. Not the PDFs or their conversions: `gbench fetch` + `gbench convert`
rebuild those byte-exactly from the pinned digests.

So every published number can be re-derived from a clone with no GPU, and every
paper reproduced exactly. Checking a `gold_quote` needs the papers, and the
linter says so — `not verified offline` — rather than passing quietly.

Set `licence_policy: redistributable` in the manifest and `convert` also writes
`sections.json`, the projected text. Then quotes are checkable from a clone
forever, and the corpus is restricted to CC-BY or better.

## Status

B0. The harness runs end to end on a generated fixture document with a scripted
model, in CI, with no network. The corpus (B1) and the two competitor adapters
(B2) are not done — `infra/engines/docling_agent.py` and `pageindex.py` are
written against the published upstream signatures and have **not been run
against an installed version**. They say so at the top of the file.
