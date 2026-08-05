# gbench — an MMLU-light for document readers

Compares [`docling-agent`](https://github.com/docling-project/docling-agent),
[PageIndex](https://github.com/VectifyAI/PageIndex) and
[`glosa`](..) on the same document, with the same model, under the same budget.

Multiple choice, scored by exact match, surrounded by controls that stop the
number lying: a **closed-book** floor, an **oracle-context** ceiling, option
**rotation** against position bias, and distractors taken from the document
itself so keyword overlap cannot win.

Design and rationale: [`../docs/EVAL.md`](../docs/EVAL.md).

## Use

```bash
uv sync                                   # glosa + pyyaml. Nothing else.
uv run gbench lint                        # keep the corpus honest — no model
uv run gbench run --engines glosa,closed-book,oracle-context
uv run gbench score --policy declared
uv run gbench score --policy strict       # re-scores the same journal
```

Competitors live behind extras, and their adapters import inside the
constructor — so the two commands above work on a machine with neither
`mellea` nor `litellm` installed:

```bash
uv sync --extra docling-agent --extra pageindex
uv run gbench run --engines glosa,docling-agent,pageindex,closed-book,oracle-context
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

## Status

B0. The harness runs end to end on a generated fixture document with a
scripted model, in CI. The corpus (B1) and the two competitor adapters (B2)
are not done — `infra/engines/docling_agent.py` and `pageindex.py` are written
against the published upstream signatures and have **not been run against an
installed version**. They say so at the top of the file.
