"""Rendering — and the two things the report is not allowed to do.

**It never prints 0 for "cannot report".** `docling-agent` does not count its
LLM calls and PageIndex has no abstention signal; both render `—`, and the
capability table below the results says why. A zero in a comparison table reads
as a measurement.

**It never ranks on a contended stopwatch.** Rows recorded at concurrency > 1
carry `contended`, and their timings are shown as `~` with a footnote rather
than as a p95 anybody could quote.

One more habit, from §5: differences are reported as paired intervals, and an
interval that spans zero is printed as `not separable` in the same breath as
the delta. A table that can only ever declare a winner will eventually declare
one that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gbench.domain.metrics import Interval

DASH = "—"


def cell(value: object, *, pct: bool = False, digits: int = 1) -> str:
    """`—` for absent, never 0."""
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not pct:
        return str(value)
    if isinstance(value, float):
        return f"{value * 100:.{digits}f} %" if pct else f"{value:.{digits}f}"
    return str(value)


@dataclass(frozen=True, slots=True)
class EngineRow:
    """One engine's line in the headline table."""

    engine: str
    accuracy: float
    lift: float | None
    headroom: float | None
    flip_rate: float | None
    unparsed_rate: float
    error_rate: float
    flip_observed: int
    """How many items were actually seen at more than one rotation.

    `flip_rate` over three items is not a rate; printing the count next to it
    is cheaper than explaining that every time."""
    abstention_recall: float | None
    false_abstention: float | None
    hit_at_1: float | None
    hit_at_3: float | None
    llm_calls_warm: float | None
    llm_calls_cold: float | None
    prompt_chars: float | None
    p50_s: float | None
    p95_s: float | None
    contended: bool
    grounded_rate: float | None
    abstention_signal: str


HEADERS: tuple[tuple[str, str], ...] = (
    ("engine", "engine"),
    ("accuracy", "acc"),
    ("lift", "lift vs closed-book"),
    ("headroom", "headroom to oracle"),
    ("hit_at_1", "hit@1"),
    ("hit_at_3", "hit@3"),
    ("flip_rate", "flip"),
    ("flip_observed", "flip n"),
    ("unparsed_rate", "unparsed"),
    ("error_rate", "error"),
    ("abstention_recall", "abst. recall"),
    ("false_abstention", "false abst."),
    ("llm_calls_warm", "calls (warm)"),
    ("llm_calls_cold", "calls (cold)"),
    ("p50_s", "p50 s"),
    ("p95_s", "p95 s"),
    ("grounded_rate", "grounded"),
)

_PCT = frozenset(
    {
        "accuracy",
        "lift",
        "headroom",
        "hit_at_1",
        "hit_at_3",
        "flip_rate",
        "unparsed_rate",
        "error_rate",
        "abstention_recall",
        "false_abstention",
        "grounded_rate",
    }
)


def headline(rows: Sequence[EngineRow]) -> str:
    out = ["| " + " | ".join(label for _, label in HEADERS) + " |"]
    out.append("| " + " | ".join("---" for _ in HEADERS) + " |")
    for row in rows:
        cells: list[str] = []
        for key, _ in HEADERS:
            value = getattr(row, key)
            text = cell(value, pct=key in _PCT)
            if key in {"p50_s", "p95_s"} and row.contended and value is not None:
                text = f"~{text}"
            cells.append(text)
        out.append("| " + " | ".join(cells) + " |")
    if any(row.contended for row in rows):
        out.append("")
        out.append(
            "`~` — recorded under concurrency; indicative only. Run "
            "`gbench run --latency-pass` for timings that can be quoted."
        )
    return "\n".join(out)


def capabilities(signals: Mapping[str, str]) -> str:
    """What each engine can report — printed under the table, always.

    `—` in a column above means "this engine has no such signal", and a reader
    should not have to guess which of the dashes are gaps and which are zeros.
    """
    out = ["| engine | abstention signal |", "| --- | --- |"]
    out += [f"| {engine} | {signal} |" for engine, signal in signals.items()]
    return "\n".join(out)


def breakdown(title: str, matrix: Mapping[str, Mapping[str, float | None]]) -> str:
    """Engines down the side, kinds or domains across the top.

    MMLU reports per subject rather than one average, and the reason applies
    here: an engine that is 20 points behind on `table` and level everywhere
    else has a serializer problem, not a navigation problem, and the headline
    number cannot tell you which. Columns with an `n` too small to mean
    anything are still printed — hiding them would misrepresent coverage — so
    read them against the per-column counts in the suite, not on their own.
    """
    columns = sorted({column for row in matrix.values() for column in row})
    if not columns:
        return f"### {title}\n\n(nothing recorded)"
    out = [f"### {title}", "", "| engine | " + " | ".join(columns) + " |"]
    out.append("| --- |" + " --- |" * len(columns))
    for engine in sorted(matrix):
        cells = [cell(matrix[engine].get(column), pct=True) for column in columns]
        out.append(f"| {engine} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def comparisons(deltas: Iterable[tuple[str, str, Interval]]) -> str:
    out = ["| comparison | Δ accuracy (pts) | verdict |", "| --- | --- | --- |"]
    for left, right, interval in deltas:
        verdict = "separable" if interval.separable else "**not separable**"
        out.append(f"| {left} vs {right} | {interval} | {verdict} |")
    return "\n".join(out)
