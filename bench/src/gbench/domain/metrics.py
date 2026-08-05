"""Aggregation — and the arithmetic that keeps a 200-item suite honest.

Two things here are not obvious and are the reason this is a module rather
than three list comprehensions in the report.

**An item is the unit, not a run.** Four rotations of the same question are not
four observations; they are one observation measured four times. Accuracy is
therefore the mean over *items* of the mean over rotations, and every interval
resamples items.

**Comparisons are paired.** On 200 items an accuracy near 70 % carries a
±6.4-point interval as an independent proportion — enough to separate engines
ten points apart, not four. But they answer the *same* items, so the comparison
can be made per-item, and that has far more power. `paired_delta` is the only
sanctioned way to say one engine beats another, and it is allowed to say "not
separable" (`EVAL.md` §5).

The breakdowns are weaker than the headline and must be read that way: 40 items
per kind is about ±14 points, 25 per domain about ±18. They are for finding a
hypothesis, not settling one.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gbench.domain.item import ABSTAIN_LETTER, unrotate
from gbench.domain.scoring import Outcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gbench.domain.scoring import ScoredRun

BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260805


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and a 95 % bootstrap interval, in percentage points."""

    point: float
    low: float
    high: float

    @property
    def separable(self) -> bool:
        """True when the interval excludes zero — i.e. the difference holds."""
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:+.1f} [{self.low:+.1f}, {self.high:+.1f}]"


def per_item_accuracy(runs: Iterable[ScoredRun]) -> dict[str, float]:
    """Item id → share of its rotations answered correctly."""
    tally: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        tally[run.item_id].append(1 if run.correct else 0)
    return {item: sum(hits) / len(hits) for item, hits in tally.items()}


def accuracy(runs: Iterable[ScoredRun]) -> float:
    scores = per_item_accuracy(runs)
    return sum(scores.values()) / len(scores) if scores else 0.0


def rate(runs: Iterable[ScoredRun], outcome: Outcome) -> float:
    """Share of *runs* with a given outcome. Unparsed and error are run-level
    properties — an engine that fails to emit a letter at rotation 2 and
    succeeds at rotation 3 failed once."""
    materialised = list(runs)
    if not materialised:
        return 0.0
    return sum(run.outcome is outcome for run in materialised) / len(materialised)


def flip_rate(runs: Iterable[ScoredRun]) -> float | None:
    """Share of items whose *chosen option* changes when the options move.

    Computed over the items actually seen at more than one rotation, and
    `None` when there are none. That distinction is load-bearing once the
    default plan stops sweeping every item (`item.schedule`): a suite where
    most items run at a single rotation would otherwise report every one of
    them as "did not flip" and drive the number to zero, which is not
    stability — it is not having looked.

    Unparsed runs count as their own answer: an engine that emits a letter at
    one rotation and prose at another is also unstable.
    """
    picks: dict[str, set[str]] = defaultdict(set)
    seen_rotations: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        chosen = run.chosen
        picks[run.item_id].add("?" if chosen is None else unrotate(chosen, run.rotation))
        seen_rotations[run.item_id].add(run.rotation)

    observed = [item for item, rotations in seen_rotations.items() if len(rotations) > 1]
    if not observed:
        return None
    return sum(len(picks[item]) > 1 for item in observed) / len(observed)


def flip_observed(runs: Iterable[ScoredRun]) -> int:
    """Items seen at more than one rotation — the `n` behind `flip_rate`."""
    seen: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        seen[run.item_id].add(run.rotation)
    return sum(len(rotations) > 1 for rotations in seen.values())


def abstention(runs: Iterable[ScoredRun], *, expected: Mapping[str, bool]) -> tuple[float, float]:
    """`(recall, false_abstention)` over runs.

    Recall is over the items where `E` is right; false abstention is over the
    ones where it is not. Reported as a pair because either alone is gameable:
    an engine that always abstains has perfect recall.
    """
    hit = total_expected = false = total_answerable = 0
    for run in runs:
        chose_e = run.chosen == ABSTAIN_LETTER
        if expected.get(run.item_id, False):
            total_expected += 1
            hit += chose_e
        else:
            total_answerable += 1
            false += chose_e
    return (
        hit / total_expected if total_expected else 0.0,
        false / total_answerable if total_answerable else 0.0,
    )


def hit_at_k(read_refs: Sequence[str], gold_refs: Sequence[str], k: int) -> bool:
    """Did the engine read a gold ref within its first `k` reads?

    Navigation quality, independent of the answer — the free label every item
    carries. An engine can be right without a hit (guessing that lands) and
    wrong with one (read it, misread it); those are different failures and the
    report separates them.
    """
    if not gold_refs:
        return False
    return bool(set(read_refs[:k]) & set(gold_refs))


def read_precision(
    read_refs: Sequence[str], gold_refs: Sequence[str], chars: Mapping[str, int]
) -> float | None:
    """Gold characters over characters read — the cost of over-reading.

    `None` when the engine does not report what it read, which is a capability
    gap and must never render as 0.
    """
    if not read_refs:
        return None
    total = sum(chars.get(ref, 0) for ref in read_refs)
    if total == 0:
        return None
    gold = sum(chars.get(ref, 0) for ref in read_refs if ref in set(gold_refs))
    return gold / total


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile. No numpy, no interpolation, no surprises."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def paired_delta(
    left: Iterable[ScoredRun],
    right: Iterable[ScoredRun],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """`left` minus `right` in accuracy points, with a paired bootstrap interval.

    Only items both engines answered are compared; resampling is over those
    items, so the pairing is preserved and the interval reflects the suite's
    real size rather than the run count.
    """
    a, b = per_item_accuracy(left), per_item_accuracy(right)
    shared = sorted(set(a) & set(b))
    if not shared:
        return Interval(0.0, 0.0, 0.0)

    deltas = [a[item] - b[item] for item in shared]
    point = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[min(int(0.975 * resamples), resamples - 1)]
    return Interval(point * 100, low * 100, high * 100)
