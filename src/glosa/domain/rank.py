"""Ranking retrieval units for one question.

Several lexical views of a unit disagree usefully:

* the **heading** is short and high-signal — when it is informative;
* the **body** is long and forgiving — it catches what "Article 7" hides;
* the **summary**, when the host's enrichment pipeline wrote one, says what the
  section is *about* rather than which words it happens to contain;
* and each of those again, queried with an **expansion** — retrieval vocabulary
  the model wrote for this question. That is what answers a paraphrase: the
  user asks for "le montant maximum que le fournisseur devra rembourser", the
  document says "plafond d'indemnisation", and only a model bridges the two.

They are combined by reciprocal-rank fusion, which uses only their orderings.
A view that is absent — no enrichment, no expansion — contributes nothing
instead of needing a special case, and a bad expansion can only *add*
candidates, never displace the ones the user's own words found.

The result is a shortlist plus a measure of how much to trust it. It is never
a decision on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.lexical import Bm25Index

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.domain.index import DocIndex, Unit

RRF_K = 60
"""Rank-fusion damping. 60 is the value from the original RRF paper; the
ranking is insensitive to it within an order of magnitude."""

SUMMARY_WEIGHT = 1.5
"""The summary view counts for more than a raw term match.

It is the only view written *about* the section rather than lifted from it, so
when the host paid an LLM call to produce it, it deserves more than an equal
vote. Applied to the fused rank contribution, not to a raw BM25 score, so it
stays scale-free."""

EXPANSION_WEIGHT = 0.25
"""Bounded so an expansion can rescue a near-tie but never overrule a real match.

The expansion is a *guess* about vocabulary. The guarantee that makes it safe to
enable by default is arithmetic, not hopeful: the three expansion views
contribute at most `(1.0 + 1.0 + SUMMARY_WEIGHT) x W / (RRF_K + 1)`, which at
W = 0.25 is 0.0143 — below the 0.0164 a *single* original view contributes at
rank 1. So a candidate found only by the expansion can never outrank one the
user's own words ranked first, and a candidate two original views agree on is
untouchable.

Raising this above ~0.28 breaks that property. `tests/test_packing.py` pins it."""

UNIT_SCORE = 1.0 / (RRF_K + 1)
"""What one view contributes when it ranks a unit first.

Makes the fused score legible: ~0.016 means "one view, and only just"; ~0.033
means two views agree. `Shortlist.margin` reads that agreement."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A unit proposed for reading, with why it was proposed."""

    unit: Unit
    score: float
    ranks: tuple[tuple[str, int], ...] = ()
    """`(view name, 0-based rank)` for every view that ranked this unit."""

    @property
    def ref(self) -> str:
        return self.unit.ref

    @property
    def views(self) -> int:
        """How many views agree this unit is relevant."""
        return len(self.ranks)

    @property
    def rationale(self) -> str:
        """Human-readable reason, recorded on the step when retrieval chose."""
        if not self.ranks:
            return "lexical match"
        parts = ", ".join(f"{name} rank {rank + 1}" for name, rank in self.ranks)
        return f"lexical match ({parts})"


@dataclass(frozen=True, slots=True)
class Shortlist:
    """What retrieval proposes, and how sure it is.

    `confident` is what decides whether the model gets involved. Testing the
    list for *emptiness* was the bug: a spurious match on a common word makes
    a non-empty, wrong shortlist, which silently suppressed the fallback.
    """

    candidates: tuple[Candidate, ...] = ()
    coverage: float = 0.0
    """Share of the question's answerable terms the top candidate contains."""
    margin: float = 0.0
    """Top score over runner-up. 1.0 means a flat ranking — noise."""

    def __bool__(self) -> bool:
        return bool(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> Candidate:
        return self.candidates[index]

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def confident(self, *, min_coverage: float, min_margin: float) -> bool:
        """True when the ranking is worth acting on without asking the model.

        Both signals are required because they fail differently. `margin`
        catches the spurious match — a common word puts the wrong section on
        top of a ranking that is otherwise flat. `coverage` catches the thin
        match — a five-word question where the leader contains one of them.
        Neither alone is enough: in the motivating case coverage was a perfect
        1.0 for the *wrong* section.
        """
        if not self.candidates:
            return False
        return self.coverage >= min_coverage and self.margin >= min_margin


class UnitRanker:
    """Lexical shortlist over the units of one document.

    Built once per document and reused across questions — the indexes depend on
    the document, not the query.
    """

    def __init__(self, index: DocIndex) -> None:
        self._units = {unit.ref: unit for unit in index.units}
        # Deliberately `text_of`, not `excerpt`: the excerpt is budget-capped,
        # and indexing a truncated section hides precisely the terms that make
        # a long section worth opening.
        self._body = Bm25Index((unit.ref, index.text_of(unit.ref)) for unit in index.units)
        self._titles = Bm25Index((unit.ref, unit.title) for unit in index.units)
        self._summaries = Bm25Index(
            (unit.ref, _summary_text(unit)) for unit in index.units if unit.enriched
        )

    @property
    def body(self) -> Bm25Index:
        return self._body

    @property
    def uses_enrichment(self) -> bool:
        """True when the host's pipeline gave us summaries to rank on."""
        return len(self._summaries) > 0

    def shortlist(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude: Sequence[str] = (),
        expansion: Sequence[str] = (),
    ) -> Shortlist:
        """Best units for `query`, fused across every available view."""
        skip = frozenset(exclude)
        views: list[tuple[str, Bm25Index, str, float]] = [
            ("text", self._body, query, 1.0),
            ("heading", self._titles, query, 1.0),
            ("summary", self._summaries, query, SUMMARY_WEIGHT),
        ]
        if expansion:
            expanded = " ".join(expansion)
            views += [
                ("text+", self._body, expanded, EXPANSION_WEIGHT),
                ("heading+", self._titles, expanded, EXPANSION_WEIGHT),
                ("summary+", self._summaries, expanded, SUMMARY_WEIGHT * EXPANSION_WEIGHT),
            ]

        scores: dict[str, float] = {}
        ranks: dict[str, list[tuple[str, int]]] = {}
        for name, index, text, weight in views:
            for position, hit in enumerate(index.rank(text)):
                if hit.ref in skip or hit.ref not in self._units:
                    continue
                scores[hit.ref] = scores.get(hit.ref, 0.0) + weight / (RRF_K + position + 1)
                ranks.setdefault(hit.ref, []).append((name, position))

        fused = [
            Candidate(unit=self._units[ref], score=score, ranks=tuple(ranks[ref]))
            for ref, score in scores.items()
        ]
        fused.sort(key=lambda c: (-c.score, c.unit.order))
        fused = fused[:limit]
        if not fused:
            return Shortlist()

        # Coverage is measured over everything we searched for, expansion
        # included. Measuring it on the question alone would penalise exactly
        # the case the expansion exists to rescue: the right section is the one
        # that does *not* use the question's words.
        searched = " ".join([query, *expansion])
        return Shortlist(
            candidates=tuple(fused),
            coverage=self._body.coverage(searched, [fused[0].ref]),
            margin=_margin(fused),
        )


def _margin(candidates: Sequence[Candidate]) -> float:
    """How much the leader beats the runner-up. Infinite when it stands alone."""
    if len(candidates) < 2:
        return float("inf")
    runner_up = candidates[1].score
    return float("inf") if runner_up <= 0 else candidates[0].score / runner_up


def _summary_text(unit: Unit) -> str:
    return " ".join([unit.summary, *unit.keywords]).strip()
