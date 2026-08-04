"""Ranking retrieval units for one question.

Three lexical views of a unit disagree usefully:

* the **heading** is short and high-signal — when it is informative;
* the **body** is long and forgiving — it catches what "Article 7" hides;
* the **summary**, when the host's enrichment pipeline wrote one, says what the
  section is *about* rather than which words it happens to contain. That is the
  signal PageIndex builds its entire index around, and it is the only one of
  the three that survives a paraphrased question.

Rather than tune weights between three scores on incomparable scales, they are
combined by reciprocal-rank fusion, which uses only their orderings. A view
that is absent — no enrichment, no heading — simply contributes nothing instead
of needing a special case.

The result is a shortlist, never a decision. `HybridStrategy` reads the top of
it and asks a model to navigate when it comes back empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from glosa.domain.lexical import Bm25Index, Scored

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


@dataclass(frozen=True, slots=True)
class Candidate:
    """A unit proposed for reading, with why it was proposed."""

    unit: Unit
    score: float
    body_rank: int | None = None
    title_rank: int | None = None
    summary_rank: int | None = None

    @property
    def ref(self) -> str:
        return self.unit.ref

    @property
    def rationale(self) -> str:
        """Human-readable reason, recorded on the step when retrieval chose."""
        parts = [
            f"{name} rank {rank + 1}"
            for name, rank in (
                ("summary", self.summary_rank),
                ("heading", self.title_rank),
                ("text", self.body_rank),
            )
            if rank is not None
        ]
        return "lexical match (" + ", ".join(parts) + ")" if parts else "lexical match"


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
    ) -> list[Candidate]:
        """Best units for `query`, fused across the three views, best first.

        Empty means *no lexical signal at all* — not "everything scored badly".
        The caller must treat that as "ask the model" rather than "give up".
        """
        skip = frozenset(exclude)
        body = _positions(self._body.rank(query))
        title = _positions(self._titles.rank(query))
        summary = _positions(self._summaries.rank(query))

        fused: list[Candidate] = []
        for ref in set(body) | set(title) | set(summary):
            if ref in skip or ref not in self._units:
                continue
            body_rank, title_rank, summary_rank = body.get(ref), title.get(ref), summary.get(ref)
            score = _rrf(body_rank) + _rrf(title_rank) + SUMMARY_WEIGHT * _rrf(summary_rank)
            fused.append(
                Candidate(
                    unit=self._units[ref],
                    score=score,
                    body_rank=body_rank,
                    title_rank=title_rank,
                    summary_rank=summary_rank,
                )
            )

        fused.sort(key=lambda c: (-c.score, c.unit.order))
        return fused[:limit]


def _summary_text(unit: Unit) -> str:
    return " ".join([unit.summary, *unit.keywords]).strip()


def _positions(hits: Sequence[Scored]) -> dict[str, int]:
    return {hit.ref: position for position, hit in enumerate(hits)}


def _rrf(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / (RRF_K + rank + 1)
