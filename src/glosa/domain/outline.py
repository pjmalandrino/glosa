"""Rendering the document map the model navigates by.

The upstream loop re-injects the complete outline into every selection prompt.
On a long report that outline alone can exceed the context window, and the run
fails in a way that looks like a model problem.

Here the outline is fitted to a character budget by degrading in a fixed order —
drop the deepest heading levels first, then elide the middle of the list — and
**every reduction is announced in the text itself**, so the model knows it is
looking at a partial map and glosa never presents a truncated view as complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from glosa.domain.index import Unit

DEFAULT_OUTLINE_BUDGET = 6_000
_MIN_KEPT = 8


def render_outline(
    units: Sequence[Unit],
    *,
    char_budget: int = DEFAULT_OUTLINE_BUDGET,
    visited: Collection[str] = (),
) -> str:
    """Render `units` as an indented, ref-annotated map that fits the budget."""
    if not units:
        return "(empty document)"

    seen = frozenset(visited)
    max_level = max(_level(u) for u in units)

    # 1. Drop the deepest heading levels until the map fits.
    kept: Sequence[Unit] = units
    while max_level > 0:
        text = _render(kept, seen)
        if len(text) <= char_budget:
            return text
        max_level -= 1
        deeper = [u for u in units if _level(u) <= max_level]
        if len(deeper) < _MIN_KEPT:
            break
        kept = deeper

    # 2. Still too long: elide the middle, keeping head and tail.
    text = _render(kept, seen)
    if len(text) <= char_budget:
        return text
    return _render_elided(kept, seen, char_budget)


def _render(units: Sequence[Unit], visited: frozenset[str]) -> str:
    return "\n".join(_line(u, visited) for u in units)


def _render_elided(units: Sequence[Unit], visited: frozenset[str], char_budget: int) -> str:
    head: list[str] = []
    tail: list[str] = []
    used = 0
    marker_cost = 60
    lo, hi = 0, len(units) - 1
    while lo <= hi:
        line = _line(units[lo], visited)
        if used + len(line) + marker_cost > char_budget:
            break
        head.append(line)
        used += len(line) + 1
        lo += 1
        if lo > hi:
            break
        line = _line(units[hi], visited)
        if used + len(line) + marker_cost > char_budget:
            break
        tail.append(line)
        used += len(line) + 1
        hi -= 1

    omitted = hi - lo + 1
    if omitted <= 0:
        return "\n".join(head + list(reversed(tail)))
    marker = f"  […  {omitted} section(s) omitted from this map — ask to widen the outline …]"
    return "\n".join([*head, marker, *reversed(tail)])


def _level(unit: Unit) -> int:
    """Heading level for display; unlevelled units sit at the top."""
    return 0 if unit.level is None else max(0, unit.level)


def _line(unit: Unit, visited: frozenset[str]) -> str:
    indent = "  " * _level(unit)
    mark = "✓ " if unit.ref in visited else ""
    page = f", p.{unit.page_no}" if unit.page_no is not None else ""
    return (
        f"{indent}- {mark}{unit.ref} · {unit.label}{page} · "
        f"{_human(unit.char_len)} · {unit.title or '(untitled)'}"
    )


def _human(n: int) -> str:
    if n < 1_000:
        return f"{n} chars"
    return f"{n / 1000:.1f}k chars"
