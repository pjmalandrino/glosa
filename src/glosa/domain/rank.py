"""Ranking retrieval units for one question.

Two lexical views of a unit disagree usefully: its **heading** is short and
high-signal when it is informative, its **body** is long and forgiving when the
heading is not ("Article 7"). Rather than tune a weight between two scores on
incomparable scales, they are combined by reciprocal-rank fusion, which only
uses their orderings.

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


@dataclass(frozen=True, slots=True)
class Candidate:
    """A unit proposed for reading, with why it was proposed."""

    unit: Unit
    score: float
    body_rank: int | None = None
    title_rank: int | None = None

    @property
    def ref(self) -> str:
        return self.unit.ref

    @property
    def rationale(self) -> str:
        """Human-readable reason, recorded on the step when retrieval chose."""
        parts = []
        if self.body_rank is not None:
            parts.append(f"text rank {self.body_rank + 1}")
        if self.title_rank is not None:
            parts.append(f"heading rank {self.title_rank + 1}")
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

    @property
    def body(self) -> Bm25Index:
        return self._body

    def shortlist(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude: Sequence[str] = (),
    ) -> list[Candidate]:
        """Best units for `query`, fused across heading and body, best first.

        Empty means *no lexical signal at all* — not "everything scored badly".
        The caller must treat that as "ask the model" rather than "give up".
        """
        skip = frozenset(exclude)
        body = _positions(self._body.rank(query))
        title = _positions(self._titles.rank(query))

        fused: list[Candidate] = []
        for ref in set(body) | set(title):
            if ref in skip or ref not in self._units:
                continue
            body_rank = body.get(ref)
            title_rank = title.get(ref)
            score = _rrf(body_rank) + _rrf(title_rank)
            fused.append(
                Candidate(
                    unit=self._units[ref],
                    score=score,
                    body_rank=body_rank,
                    title_rank=title_rank,
                )
            )

        fused.sort(key=lambda c: (-c.score, c.unit.order))
        return fused[:limit]


def _positions(hits: Sequence[Scored]) -> dict[str, int]:
    return {hit.ref: position for position, hit in enumerate(hits)}


def _rrf(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / (RRF_K + rank + 1)
