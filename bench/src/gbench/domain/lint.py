"""The linter that makes a hand-written suite trustworthy.

We are writing the benchmark our own engine is measured on. That is a real
conflict of interest, and the only honest response is to make the ways of
cheating mechanically visible rather than to promise we did not.

Each check below corresponds to a way a light MCQ suite goes bad:

* a *gold quote* that is not in the document means the answer was authored from
  memory, or the ref is wrong;
* a *distractor* that appears nowhere in the document was invented, and an
  invented distractor is implausible, and an implausible distractor turns the
  item into a common-sense question with a 20 % floor;
* a distractor sitting *inside* a gold ref makes two options defensible;
* an *unbalanced key* lets an engine with a letter preference score above
  chance;
* the *longest option* being right more often than chance is the oldest MCQ
  artefact there is, and small models exploit it;
* an item answerable *closed-book* was never about the document.

Pure: the checks take text through callables, so nothing here knows what a
document is made of.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from gbench.domain.item import ABSTAIN_LETTER, LETTERS, ItemKind
from gbench.domain.scoring import fold, letter_distribution

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from gbench.domain.item import Item

CHI2_CRIT_DF4_P05 = 9.488
"""χ² critical value, 4 degrees of freedom (five letters), alpha = 0.05.

Hardcoded rather than pulled from `scipy`: one constant is not worth a
dependency that drags numpy into a linter."""

Z_CRIT = 2.0
"""approx. alpha 0.05, two-sided. Used for the longest-option artefact."""


class Severity(StrEnum):
    ERROR = "error"
    """The suite is unusable until fixed."""
    WARN = "warn"
    """Reported, does not fail the build — small-sample noise can trip these."""


@dataclass(frozen=True, slots=True)
class Finding:
    severity: Severity
    check: str
    item_id: str
    message: str

    def __str__(self) -> str:
        where = f" [{self.item_id}]" if self.item_id else ""
        return f"{self.severity.value}: {self.check}{where}: {self.message}"


TextOf = Callable[[str, str], str]
"""(doc slug, ref) → the text of that node. Empty string when the ref is
unknown — the linter reports that rather than raising."""

DocText = Callable[[str], str]
"""doc slug → the whole document's text."""


def lint(
    items: Sequence[Item],
    *,
    text_of: TextOf,
    doc_text: DocText,
    leaky: Iterable[str] = (),
) -> list[Finding]:
    """Every check, in one pass. Empty list means the suite is publishable."""
    findings: list[Finding] = []
    findings += _check_provenance(items)
    findings += _check_gold_quotes(items, text_of)
    findings += _check_distractors(items, text_of, doc_text)
    findings += _check_not_stated(items, text_of)
    findings += _check_key_balance(items)
    findings += _check_longest_option(items)
    findings += _check_leaky(items, leaky)
    return findings


def _check_provenance(items: Sequence[Item]) -> list[Finding]:
    out: list[Finding] = []
    for item in items:
        if item.authored_from == "trace":
            out.append(
                Finding(
                    Severity.ERROR,
                    "provenance",
                    item.id,
                    "authored from a trace — an item written by watching an engine "
                    "read is an item that engine will pass",
                )
            )
        if not item.authored_by:
            out.append(Finding(Severity.WARN, "provenance", item.id, "no author recorded"))
    return out


def _check_gold_quotes(items: Sequence[Item], text_of: TextOf) -> list[Finding]:
    out: list[Finding] = []
    for item in items:
        if item.kind is ItemKind.NOT_STATED:
            continue
        if not item.gold_refs:
            out.append(Finding(Severity.ERROR, "gold_refs", item.id, "no gold ref"))
            continue
        haystack = fold(" ".join(text_of(item.doc, ref) for ref in item.gold_refs))
        if not haystack:
            out.append(
                Finding(
                    Severity.ERROR,
                    "gold_refs",
                    item.id,
                    f"refs resolve to no text: {list(item.gold_refs)}",
                )
            )
        elif item.gold_quote and fold(item.gold_quote) not in haystack:
            out.append(
                Finding(
                    Severity.ERROR,
                    "gold_quote",
                    item.id,
                    f"not found in {list(item.gold_refs)}: {item.gold_quote[:60]!r}",
                )
            )
        elif not item.gold_quote:
            out.append(Finding(Severity.WARN, "gold_quote", item.id, "no quote to verify"))
    return out


def _check_distractors(items: Sequence[Item], text_of: TextOf, doc_text: DocText) -> list[Finding]:
    out: list[Finding] = []
    for item in items:
        whole = fold(doc_text(item.doc))
        gold_text = fold(" ".join(text_of(item.doc, ref) for ref in item.gold_refs))
        for letter in LETTERS:
            if letter == item.answer:
                continue
            option = fold(item.options[letter])
            if len(option) < 4:
                continue
            if whole and option not in whole:
                out.append(
                    Finding(
                        Severity.WARN,
                        "distractor_source",
                        item.id,
                        f"option {letter} appears nowhere in the document — invented "
                        f"distractors make the item a common-sense question",
                    )
                )
            if gold_text and option in gold_text:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "distractor_overlap",
                        item.id,
                        f"option {letter} sits inside a gold ref — two defensible answers",
                    )
                )
            ref = dict(item.distractor_refs).get(letter, "")
            if ref and ref in item.gold_refs:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "distractor_overlap",
                        item.id,
                        f"option {letter} is sourced from a gold ref",
                    )
                )
    return out


def _check_not_stated(items: Sequence[Item], text_of: TextOf) -> list[Finding]:
    """The removal has to be auditable, and it has to be checked in one place.

    Not the whole document: a `not_stated` item's distractors are still drawn
    from elsewhere in it (§3.4), so "no option appears anywhere" would forbid
    exactly the plausible distractors the format needs. What must hold is
    narrower and checkable — the section the value was removed from no longer
    contains any of them.
    """
    out: list[Finding] = []
    for item in items:
        if item.kind is not ItemKind.NOT_STATED:
            continue
        if not item.removed_from:
            out.append(
                Finding(
                    Severity.ERROR,
                    "not_stated",
                    item.id,
                    "no `removed_from` ref — the removal must be auditable",
                )
            )
            continue
        section = fold(text_of(item.doc, item.removed_from))
        if not section:
            out.append(
                Finding(
                    Severity.ERROR,
                    "not_stated",
                    item.id,
                    f"`removed_from` ref does not resolve: {item.removed_from}",
                )
            )
            continue
        for letter in LETTERS:
            option = fold(item.options[letter])
            if len(option) >= 8 and option in section:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "not_stated",
                        item.id,
                        f"option {letter} is still in {item.removed_from} — "
                        f"then E is not the answer",
                    )
                )
    return out


def _check_key_balance(items: Sequence[Item]) -> list[Finding]:
    if len(items) < 20:
        return []
    counts = letter_distribution([item.answer for item in items])
    expected = len(items) / len(counts)
    chi2 = sum((observed - expected) ** 2 / expected for observed in counts.values())
    if chi2 > CHI2_CRIT_DF4_P05:
        return [
            Finding(
                Severity.WARN,
                "key_balance",
                "",
                f"answer key is skewed (χ²={chi2:.1f} > {CHI2_CRIT_DF4_P05}): {counts}",
            )
        ]
    return []


def _check_longest_option(items: Sequence[Item]) -> list[Finding]:
    """The correct answer being the longest option more often than chance.

    Checked over answerable items only: on a `not_stated` item the answer is the
    fixed sentinel, whose length says nothing about the author.
    """
    answerable = [item for item in items if item.answer != ABSTAIN_LETTER]
    if len(answerable) < 20:
        return []
    longest = 0
    for item in answerable:
        lengths = {letter: len(item.options[letter]) for letter in LETTERS}
        top = max(lengths.values())
        if lengths[item.answer] == top and list(lengths.values()).count(top) == 1:
            longest += 1
    n = len(answerable)
    p = 1 / len(LETTERS)
    z = (longest / n - p) / ((p * (1 - p) / n) ** 0.5)
    if z > Z_CRIT:
        return [
            Finding(
                Severity.WARN,
                "longest_option",
                "",
                f"the right answer is the longest option in {longest}/{n} items "
                f"(z={z:.1f}) — a model can score above chance on length alone",
            )
        ]
    return []


def _check_leaky(items: Sequence[Item], leaky: Iterable[str]) -> list[Finding]:
    """Items the closed-book control answered correctly.

    A warning, not an error: one closed-book hit in four rotations is luck at a
    20 % floor. The scorer excludes them from the headline either way.
    """
    flagged = set(leaky)
    return [
        Finding(
            Severity.WARN,
            "leaky",
            item.id,
            "answered closed-book — excluded from the headline accuracy",
        )
        for item in items
        if item.id in flagged
    ]


def failed(findings: Iterable[Finding]) -> bool:
    return any(finding.severity is Severity.ERROR for finding in findings)
