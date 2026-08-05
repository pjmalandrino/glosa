# Picking the forty papers

Eight themes, five papers each. About thirty minutes of clicking, then one
command.

## Why these eight

Not for subject variety. The per-domain column is only worth printing if the
fields **write differently** — a maths paper puts its content in numbered
theorems and almost never in a table, an econometrics paper puts nearly all of
it in one enormous table whose columns are defined in a footnote. Those two
fail a reader in different places, and that is the whole point of the
breakdown.

Each theme is picked for the item kind it makes *hard*, so the suite has papers
where each kind is genuinely difficult rather than five easy ones and a token
one:

| # | Theme | arXiv listing | What it stresses |
| --- | --- | --- | --- |
| 1 | NLP / language models | [`cs.CL`](https://arxiv.org/list/cs.CL/recent) | `table` — dense result grids, ablations, appendix hyperparameters. The kind everyone assumes is easy |
| 2 | Statistical theory | [`math.ST`](https://arxiv.org/list/math.ST/recent) | `crossref` — "by Lemma 3.2", "under Assumption (A2)". Native multi-hop, and almost no tables at all |
| 3 | Neuroscience | [`q-bio.NC`](https://arxiv.org/list/q-bio.NC/recent) | `peripheral` — the result is in the figure caption, not the body. A heading-walker never arrives |
| 4 | Condensed matter | [`cond-mat.mtrl-sci`](https://arxiv.org/list/cond-mat.mtrl-sci/recent) | `lookup` with units and equations — a value is only right with its unit, and the text is dense with symbols |
| 5 | Econometrics | [`econ.EM`](https://arxiv.org/list/econ.EM/recent) | `table` + `crossref` — one huge regression table, columns defined in a footnote elsewhere |
| 6 | Signal processing | [`eess.SP`](https://arxiv.org/list/eess.SP/recent) | `peripheral` — algorithm blocks and pseudo-code, which conversions handle badly and which no heading owns |
| 7 | Climate / earth science | [`physics.ao-ph`](https://arxiv.org/list/physics.ao-ph/recent) | `not_stated` — long methods and data-availability sections make "is this stated anywhere?" genuinely hard |
| 8 | Software / systems | [`cs.SE`](https://arxiv.org/list/cs.SE/recent) | `lookup` in prose plus long appendices — closest to the documents Studio users actually convert |

Swap any of them. The rule that matters is not the list — it is that the eight
differ in *where the answer lives*, because that is what the engines differ in.

## What to look for in a paper

- **10–30 pages.** Under about eight pages the whole thing fits in one context
  window and nothing is being navigated — glosa's own `direct_char_threshold`
  would skip its reading loop entirely, and the bench would measure nothing.
- **At least one real results table**, or theme 1 and 5 have no `table` item.
- **An appendix, or figures with substantive captions**, or there is no
  `peripheral` item to write.
- **Posted after the model's training cutoff.** This is the one criterion with
  no wiggle room: pick from the *recent* listing, not from what you remember
  being good. A paper you already know is a paper the model already knows, and
  the closed-book control will show it as a leak.
- **Licence: don't check it.** Under the default `fetch-only` policy the corpus
  commits hashes rather than the paper's text, so any licence works. `gbench
  fetch` records what it can read off the abstract page, and a blank one is a
  warning, not a rejection.

## The list format

Write `corpus/papers.txt` — one paper per line, bare id or URL of either kind,
optional category after it, `#` comments so the themes stay next to the papers
that fill them:

```
# 1 — NLP: dense result tables
2603.01234  cs.CL
https://arxiv.org/abs/2603.05678  cs.CL
...

# 2 — statistical theory: cross-references everywhere
arxiv.org/pdf/2603.09999v2  math.ST
...
```

Then:

```bash
uv run gbench manifest --from-list corpus/papers.txt
uv run gbench fetch      # pins each PDF's SHA-256, fills licences where readable
uv run gbench convert    # docling.json, doc.md, refs.json
uv run gbench lint --partial   # structure only — no items written yet
```

The category column is what the per-domain breakdown groups by. Leave it off
and the papers all land in one column, which still works and tells you less.

## Then the questions

Five per paper, one of each kind — the quota is enforced by `gbench lint`, and
`../../docs/EVAL.md` §2 says what each kind is and §6 lists what the linter
rejects. Budget an hour per five papers; the `crossref` and `not_stated` items
are the slow ones, because both require having actually read the paper.

When the items are written:

```bash
uv run gbench lint --verify-quotes    # now that the papers are converted
```
