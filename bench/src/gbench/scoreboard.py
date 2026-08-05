"""Journal rows → the table.

The application service on the reading side, mirroring `runner.py` on the
writing side. No model is invoked here, ever: this is the half of the bench
that can be re-run by anyone with the repository and no GPU, which is what
makes a published number checkable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gbench.domain import metrics
from gbench.domain.item import ItemKind
from gbench.domain.report import EngineRow
from gbench.domain.scoring import Outcome, Policy, ScoredRun, score

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gbench.domain.item import Item
    from gbench.infra.journal import Row

CLOSED_BOOK = "closed-book"
ABSTRACT_ONLY = "abstract-only"
ORACLE = "oracle-context"

NO_NAVIGATION = (CLOSED_BOOK, ABSTRACT_ONLY)
"""Controls whose success means the item did not require reading the paper."""

CONTROLS = frozenset({CLOSED_BOOK, ABSTRACT_ONLY, ORACLE})


@dataclass(frozen=True, slots=True)
class Scoreboard:
    rows: tuple[EngineRow, ...]
    deltas: tuple[tuple[str, str, metrics.Interval], ...]
    signals: Mapping[str, str]
    leaky: tuple[str, ...]
    """Items a no-navigation control got right — excluded from the headline.

    Both the closed-book and the abstract-only controls feed this: an item the
    model answers from its weights and an item answerable from the paper's own
    summary are the same problem, which is that no reading was required."""
    policy: Policy
    by_kind: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    by_domain: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    """Items per kind and per domain, so a breakdown cell can be read against
    the `n` behind it."""


def build(
    journal_rows: Iterable[Row],
    items: Sequence[Item],
    *,
    policy: Policy = Policy.DECLARED,
    signals: Mapping[str, str] | None = None,
    exclude_leaky: bool = True,
) -> Scoreboard:
    by_id = {item.id: item for item in items}
    rendered_cache: dict[tuple[str, int], object] = {}
    scored: dict[str, list[ScoredRun]] = defaultdict(list)
    raw: dict[str, list[Row]] = defaultdict(list)

    for row in journal_rows:
        item = by_id.get(row.item_id)
        if item is None:
            continue
        key = (row.item_id, row.rotation)
        if key not in rendered_cache:
            rendered_cache[key] = item.render(row.rotation)
        rendered = rendered_cache[key]
        scored[row.engine].append(
            score(
                rendered,  # type: ignore[arg-type]
                engine=row.engine,
                text=row.text,
                abstained=row.abstained,
                errored=bool(row.error),
                policy=policy,
            )
        )
        raw[row.engine].append(row)

    leaked: set[str] = set()
    for control in NO_NAVIGATION:
        leaked |= {
            item_id
            for item_id, accuracy in metrics.per_item_accuracy(scored.get(control, ())).items()
            if accuracy > 0.5
        }
    leaky = tuple(sorted(leaked))
    drop = set(leaky) if exclude_leaky else set()

    expected = {item.id: item.kind is ItemKind.NOT_STATED for item in items}
    gold_refs = {item.id: item.gold_refs for item in items}

    closed_book = _accuracy(scored.get(CLOSED_BOOK, ()), drop)
    oracle = _accuracy(scored.get(ORACLE, ()), drop)

    out: list[EngineRow] = []
    by_kind: dict[str, Mapping[str, float | None]] = {}
    by_domain: dict[str, Mapping[str, float | None]] = {}
    for engine in sorted(scored):
        runs = [run for run in scored[engine] if run.item_id not in drop]
        engine_rows = [row for row in raw[engine] if row.item_id not in drop]
        recall, false_abstention = metrics.abstention(runs, expected=expected)
        timings = [row.wall_s for row in engine_rows]
        contended = any(row.contended for row in engine_rows)
        accuracy = metrics.accuracy(runs)
        by_kind[engine] = _slice(runs, {item.id: str(item.kind) for item in items})
        by_domain[engine] = _slice(runs, {item.id: item.domain or "—" for item in items})
        out.append(
            EngineRow(
                engine=engine,
                accuracy=accuracy,
                lift=None if engine in CONTROLS else accuracy - closed_book,
                headroom=None if engine in CONTROLS else oracle - accuracy,
                flip_rate=metrics.flip_rate(runs),
                flip_observed=metrics.flip_observed(runs),
                unparsed_rate=metrics.rate(runs, Outcome.UNPARSED),
                error_rate=metrics.rate(runs, Outcome.ERROR),
                abstention_recall=recall,
                false_abstention=false_abstention,
                hit_at_1=_hit(engine_rows, gold_refs, 1),
                hit_at_3=_hit(engine_rows, gold_refs, 3),
                llm_calls_warm=_mean(row.llm_calls for row in engine_rows),
                llm_calls_cold=_cold_calls(engine_rows),
                prompt_chars=_mean(row.prompt_chars for row in engine_rows),
                p50_s=metrics.percentile(timings, 0.5),
                p95_s=metrics.percentile(timings, 0.95),
                contended=contended,
                grounded_rate=_grounded(engine_rows),
                abstention_signal=(signals or {}).get(
                    engine, engine_rows[0].signal if engine_rows else "—"
                ),
            )
        )

    engines = [row.engine for row in out if row.engine not in CONTROLS]
    deltas = tuple(
        (left, right, metrics.paired_delta(scored[left], scored[right]))
        for index, left in enumerate(engines)
        for right in engines[index + 1 :]
    )
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        if item.id in drop:
            continue
        counts[str(item.kind)] += 1
        counts[item.domain or "—"] += 1

    return Scoreboard(
        rows=tuple(out),
        deltas=deltas,
        signals=signals or {},
        leaky=leaky,
        policy=policy,
        by_kind=by_kind,
        by_domain=by_domain,
        counts=dict(counts),
    )


def _accuracy(runs: Iterable[ScoredRun], drop: set[str]) -> float:
    return metrics.accuracy([run for run in runs if run.item_id not in drop])


def _slice(runs: Iterable[ScoredRun], label: Mapping[str, str]) -> dict[str, float | None]:
    """Accuracy per label — per kind, per domain.

    Averaged over items, like the headline: four rotations of one question are
    one observation, and a breakdown that forgot that would disagree with the
    number above it.
    """
    per_item = metrics.per_item_accuracy(runs)
    grouped: dict[str, list[float]] = defaultdict(list)
    for item_id, accuracy in per_item.items():
        grouped[label.get(item_id, "—")].append(accuracy)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _mean(values: Iterable[int | None]) -> float | None:
    """`None` when the engine does not report the quantity — never 0."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _cold_calls(rows: Sequence[Row]) -> float | None:
    """Warm cost plus this document's share of the one-off indexing cost.

    PageIndex pays an LLM call per node before the first question; amortising
    it over the items of its own document is the only way to put it in the same
    column as a per-query cost, and showing it beside the warm number is the
    only way to keep it honest."""
    warm = _mean(row.llm_calls for row in rows)
    if warm is None:
        return None
    per_doc: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.doc] += 1
        prepare = row.prepare.get("llm_calls")
        if prepare and not row.prepare.get("cached"):
            per_doc[row.doc] = int(prepare)
    if not per_doc:
        return warm
    extra = sum(per_doc[doc] / counts[doc] for doc in per_doc) / len(counts)
    return warm + extra


def _hit(rows: Sequence[Row], gold: Mapping[str, tuple[str, ...]], k: int) -> float | None:
    usable = [row for row in rows if row.read_refs]
    if not usable:
        return None
    hits = sum(metrics.hit_at_k(row.read_refs, gold.get(row.item_id, ()), k) for row in usable)
    return hits / len(usable)


def _grounded(rows: Sequence[Row]) -> float | None:
    verdicts = [row.grounded for row in rows if row.grounded is not None]
    return sum(verdicts) / len(verdicts) if verdicts else None
