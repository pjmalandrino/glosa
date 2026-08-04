"""Rendering the document map the model navigates by.

The upstream loop re-injects the complete outline into every selection prompt.
On a long report that outline alone can exceed the context window, and the run
fails in a way that looks like a model problem.

Here the outline is fitted to a character budget by degrading in a fixed order —
drop the deepest heading levels first, then elide the middle of the list — and
**every reduction is announced in the text itself**, so the model knows it is
looking at a partial map.

Rendering also returns *which refs it actually showed*. That closes a real hole:
telling a model "choose one of these 300 refs" while showing it 40 invites it to
name something it cannot see. Callers restrict the candidate list to what was
rendered, and descend into the elided children afterwards if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from glosa.domain.index import Unit

DEFAULT_OUTLINE_BUDGET = 6_000
SUMMARY_CLIP = 140
_MIN_KEPT = 8


@dataclass(frozen=True, slots=True)
class Outline:
    """A rendered map, and the refs it is honest about having shown."""

    text: str
    refs: frozenset[str] = field(default_factory=frozenset)
    complete: bool = True
    """False when levels were dropped or entries elided to fit the budget."""

    def __str__(self) -> str:
        return self.text


def render_outline(
    units: Sequence[Unit],
    *,
    char_budget: int = DEFAULT_OUTLINE_BUDGET,
    visited: Collection[str] = (),
) -> Outline:
    """Render `units` as an indented, ref-annotated map that fits the budget."""
    if not units:
        return Outline("(empty document)", frozenset(), complete=True)

    seen = frozenset(visited)
    max_level = max(_level(u) for u in units)

    # 1. Drop the deepest heading levels until the map fits.
    kept: Sequence[Unit] = units
    complete = True
    while max_level > 0:
        text = _render(kept, seen)
        if len(text) <= char_budget:
            return Outline(text, _refs(kept), complete=complete)
        max_level -= 1
        deeper = [u for u in units if _level(u) <= max_level]
        if len(deeper) < _MIN_KEPT:
            break
        kept, complete = deeper, False

    # 2. Still too long: elide the middle, keeping head and tail.
    text = _render(kept, seen)
    if len(text) <= char_budget:
        return Outline(text, _refs(kept), complete=complete)
    return _render_elided(kept, seen, char_budget)


def _render(units: Sequence[Unit], visited: frozenset[str]) -> str:
    return "\n".join(_line(u, visited) for u in units)


def _refs(units: Sequence[Unit]) -> frozenset[str]:
    return frozenset(u.ref for u in units)


def _render_elided(units: Sequence[Unit], visited: frozenset[str], char_budget: int) -> Outline:
    head: list[str] = []
    tail: list[str] = []
    shown: list[Unit] = []
    used = 0
    marker_cost = 60
    lo, hi = 0, len(units) - 1
    while lo <= hi:
        line = _line(units[lo], visited)
        if used + len(line) + marker_cost > char_budget:
            break
        head.append(line)
        shown.append(units[lo])
        used += len(line) + 1
        lo += 1
        if lo > hi:
            break
        line = _line(units[hi], visited)
        if used + len(line) + marker_cost > char_budget:
            break
        tail.append(line)
        shown.append(units[hi])
        used += len(line) + 1
        hi -= 1

    omitted = hi - lo + 1
    if omitted <= 0:
        return Outline("\n".join(head + list(reversed(tail))), _refs(shown), complete=True)
    marker = f"  […  {omitted} section(s) omitted from this map — ask to widen the outline …]"
    return Outline("\n".join([*head, marker, *reversed(tail)]), _refs(shown), complete=False)


def _level(unit: Unit) -> int:
    """Heading level for display; unlevelled units sit at the top."""
    return 0 if unit.level is None else max(0, unit.level)


def _line(unit: Unit, visited: frozenset[str]) -> str:
    indent = "  " * _level(unit)
    mark = "✓ " if unit.ref in visited else ""
    page = f", p.{unit.page_no}" if unit.page_no is not None else ""
    line = (
        f"{indent}- {mark}{unit.ref} · {unit.label}{page} · "
        f"{_human(unit.char_len)} · {unit.title or '(untitled)'}"
    )
    gist = _gist(unit)
    return f"{line}\n{indent}    ↳ {gist}" if gist else line


def _gist(unit: Unit) -> str:
    """What the host's enrichment says this section is about, if anything.

    A heading says "Article 12"; a summary says what Article 12 does. When the
    pipeline produced one, showing it is the difference between navigating a
    table of contents and navigating the document.
    """
    if unit.summary:
        return _clip(unit.summary, SUMMARY_CLIP)
    if unit.keywords:
        return _clip(", ".join(unit.keywords), SUMMARY_CLIP)
    return ""


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _human(n: int) -> str:
    if n < 1_000:
        return f"{n} chars"
    return f"{n / 1000:.1f}k chars"
