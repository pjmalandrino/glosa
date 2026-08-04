"""Rendering the document map the model navigates by.

The upstream loop re-injects the complete outline into every selection prompt.
On a long report that outline alone can exceed the context window, and the run
fails in a way that looks like a model problem.

Here the outline is fitted to a character budget by degrading in a fixed order —
drop the per-section leads first, then the deepest heading levels, then elide
the middle of the list — and **every reduction is announced in the text
itself**, so the model knows it is looking at a partial map.

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
    """Render `units` as an indented, ref-annotated map that fits the budget.

    Leads are the first thing sacrificed. Describing a section costs roughly as
    much as listing three, and the candidate list is restricted to the refs the
    map actually showed — so a richer map the model can only choose a third of
    the sections from is worse than a bare one covering all of them. Showing
    every section beats describing a few.
    """
    if not units:
        return Outline("(empty document)", frozenset(), complete=True)

    seen = frozenset(visited)
    rich = _fit(units, seen, char_budget, leads=True)
    if rich.complete or not any(_shows_lead(u, seen) for u in units):
        return rich
    plain = _fit(units, seen, char_budget, leads=False)
    return plain if len(plain.refs) > len(rich.refs) else rich


def _fit(units: Sequence[Unit], seen: frozenset[str], char_budget: int, *, leads: bool) -> Outline:
    max_level = max(_level(u) for u in units)

    # 1. Drop the deepest heading levels until the map fits.
    kept: Sequence[Unit] = units
    complete = True
    while max_level > 0:
        text = _render(kept, seen, leads)
        if len(text) <= char_budget:
            return Outline(text, _refs(kept), complete=complete)
        max_level -= 1
        deeper = [u for u in units if _level(u) <= max_level]
        if len(deeper) < _MIN_KEPT:
            break
        kept, complete = deeper, False

    # 2. Still too long: elide the middle, keeping head and tail.
    text = _render(kept, seen, leads)
    if len(text) <= char_budget:
        return Outline(text, _refs(kept), complete=complete)
    return _render_elided(kept, seen, char_budget, leads)


def _render(units: Sequence[Unit], visited: frozenset[str], leads: bool) -> str:
    return "\n".join(_line(u, visited, leads) for u in units)


def _refs(units: Sequence[Unit]) -> frozenset[str]:
    return frozenset(u.ref for u in units)


def _render_elided(
    units: Sequence[Unit], visited: frozenset[str], char_budget: int, leads: bool
) -> Outline:
    head: list[str] = []
    tail: list[str] = []
    shown: list[Unit] = []
    used = 0
    marker_cost = 60
    lo, hi = 0, len(units) - 1
    while lo <= hi:
        line = _line(units[lo], visited, leads)
        if used + len(line) + marker_cost > char_budget:
            break
        head.append(line)
        shown.append(units[lo])
        used += len(line) + 1
        lo += 1
        if lo > hi:
            break
        line = _line(units[hi], visited, leads)
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


def _line(unit: Unit, visited: frozenset[str], leads: bool) -> str:
    indent = "  " * _level(unit)
    mark = "✓ " if unit.ref in visited else ""
    page = f", p.{unit.page_no}" if unit.page_no is not None else ""
    line = (
        f"{indent}- {mark}{unit.ref} · {unit.label}{page} · "
        f"{_human(unit.char_len)} · {unit.title or '(untitled)'}"
    )
    gist = _gist(unit, _shows_lead(unit, visited) and leads)
    return f"{line}\n{indent}    ↳ {gist}" if gist else line


def _shows_lead(unit: Unit, visited: frozenset[str]) -> bool:
    """Whether this unit's lead would be rendered at all.

    Not for a unit already read: the prompt carries the reader's own note about
    it, so its opening line would be the same section described twice. The lead
    is there to help choose what has *not* been opened yet, and dropping it on
    the way past keeps the map from growing with the run.
    """
    return bool(unit.lead) and not unit.summary and unit.ref not in visited


def _gist(unit: Unit, leads: bool) -> str:
    """One line saying what this section is about. Best available source wins.

    A heading says "Article 12"; the gist says what Article 12 does. Three
    sources, in descending order of how much they were paid for:

    * the host's **summary** — written *about* the section by a model, the only
      one that can say "this clause caps liability" when the text never uses
      the word "cap";
    * the section's own **lead** — its opening sentences, free, and the
      document's own words. Only shown when it distinguishes the section from
      its neighbours (`DocIndex` drops the rest);
    * the **keywords** — a bag of terms, better than nothing.

    None of them feeds retrieval: the lead is already inside the body index, so
    it changes navigation, not ranking. Navigation is where numbered headings
    fail.
    """
    if unit.summary:
        return _clip(unit.summary, SUMMARY_CLIP)
    if leads and unit.lead:
        return _clip(unit.lead, SUMMARY_CLIP)
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
