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

from gbench.domain.item import ABSTAIN_LETTER, LETTERS, PER_PAPER, ItemKind
from gbench.domain.scoring import fold, letter_distribution

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gbench.domain.item import Item

REDISTRIBUTABLE = ("creativecommons.org/licenses/by", "creativecommons.org/publicdomain", "cc0")
"""Licence substrings that permit committing the paper's text.

CC-BY, CC-BY-SA and CC0. arXiv's default non-exclusive licence is deliberately
absent — it lets arXiv distribute the paper, not us."""


class LicencePolicy(StrEnum):
    """What the corpus commits, which decides which papers it may contain.

    The two are the same question. Restricting the corpus to CC-BY was never
    about licences for their own sake — it was the price of committing each
    paper's text so a third party could check every quote with nothing
    downloaded. Change what is committed and the restriction goes with it.
    """

    FETCH_ONLY = "fetch-only"
    """Commit refs, character counts and hashes — not the paper's text.

    Any licence is then usable, because nothing of the paper is redistributed
    beyond the sentence each item quotes, which is a citation. The corpus is
    reproduced by `gbench fetch` against a pinned SHA-256, so it is still
    exactly reproducible; it is no longer *offline*-auditable.

    The cost, stated rather than buried: verifying that a `gold_quote` really
    is in its section now requires fetching the papers first. `lint` says
    "not verified offline" instead of silently passing, and
    `lint --verify-quotes` refuses to be run without the text."""

    REDISTRIBUTABLE = "redistributable"
    """Commit the projected text too. CC-BY, CC-BY-SA or CC0 only.

    Strictly better when it is available — every quote checkable by anyone,
    forever, with a clone and no network. It just rules out most of arXiv."""


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

AbstractRefs = Callable[[str], "Sequence[str]"]
"""doc slug → the refs that make up its title and abstract."""


def lint(
    items: Sequence[Item],
    *,
    text_of: TextOf,
    doc_text: DocText,
    abstract_refs: AbstractRefs | None = None,
    licenses: Mapping[str, str] | None = None,
    policy: LicencePolicy = LicencePolicy.FETCH_ONLY,
    has_text: Callable[[str], bool] | None = None,
    leaky: Iterable[str] = (),
    require_quota: bool = True,
) -> list[Finding]:
    """Every check, in one pass. Empty list means the suite is publishable.

    `has_text` splits the checks in two. The ref-level ones — quota, licence,
    abstract leak, answer-key balance, provenance — need only the corpus's own
    metadata and always run. The text-level ones need the papers, which under
    `fetch-only` are not committed; those defer with a finding that says so
    rather than passing silently or failing on a file that was never meant to
    be there.
    """
    findings: list[Finding] = []
    readable = has_text or (lambda _: True)
    findings += _check_provenance(items)
    findings += _check_gold_quotes(items, text_of, readable)
    findings += _check_distractors(items, text_of, doc_text, readable)
    findings += _check_not_stated(items, text_of, readable)
    findings += _check_key_balance(items)
    findings += _check_longest_option(items)
    findings += _check_leaky(items, leaky)
    findings += _check_quota(items, require_quota=require_quota)
    if abstract_refs is not None:
        findings += _check_abstract(items, abstract_refs)
    if licenses is not None:
        findings += _check_licences(items, licenses, policy)
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


def _check_gold_quotes(
    items: Sequence[Item], text_of: TextOf, readable: Callable[[str], bool]
) -> list[Finding]:
    out: list[Finding] = []
    for item in items:
        if item.kind is ItemKind.NOT_STATED:
            continue
        if not item.gold_refs:
            out.append(Finding(Severity.ERROR, "gold_refs", item.id, "no gold ref"))
            continue
        if not readable(item.doc):
            # Not a pass. The claim is unchecked, and the report says which.
            out.append(
                Finding(
                    Severity.WARN,
                    "gold_quote",
                    item.id,
                    "not verified offline — run `gbench fetch && gbench convert`, "
                    "then `gbench lint --verify-quotes`",
                )
            )
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


VALUE_SHAPED = frozenset({ItemKind.LOOKUP, ItemKind.TABLE, ItemKind.NOT_STATED})
"""Kinds whose options are values — a rate, an amount, a date.

A `crossref` or `peripheral` item asks something a value cannot answer, so its
options are clauses the author recombined rather than spans lifted out of the
text. Demanding a verbatim source from those would produce a warning on every
one of them, and a check that always fires is a check everybody learns to
ignore."""


def _check_distractors(
    items: Sequence[Item], text_of: TextOf, doc_text: DocText, readable: Callable[[str], bool]
) -> list[Finding]:
    """Two different questions, and only one of them is about the document.

    *Did the author invent this option?* — checked verbatim, and only where the
    options are values.

    *Did the author lie about where it came from?* — checked wherever a
    `distractor_refs` entry claims a source. That one is an error at any kind:
    an unverifiable provenance note is worse than none, because it looks like
    evidence.
    """
    out: list[Finding] = []
    for item in items:
        if not readable(item.doc):
            continue
        whole = fold(doc_text(item.doc))
        gold_text = fold(" ".join(text_of(item.doc, ref) for ref in item.gold_refs))
        claimed = dict(item.distractor_refs)

        for letter in LETTERS:
            if letter == item.answer:
                continue
            option = fold(item.options[letter])
            if len(option) < 4:
                continue

            ref = claimed.get(letter, "")
            if ref:
                if option not in fold(text_of(item.doc, ref)):
                    out.append(
                        Finding(
                            Severity.ERROR,
                            "distractor_source",
                            item.id,
                            f"option {letter} claims to come from {ref} and is not in it",
                        )
                    )
            elif item.kind in VALUE_SHAPED and whole and option not in whole:
                out.append(
                    Finding(
                        Severity.WARN,
                        "distractor_source",
                        item.id,
                        f"option {letter} appears nowhere in the document — an invented "
                        f"value turns the item into a common-sense question",
                    )
                )

            # On a table item the distractors are *other rows of the same
            # table*, and that is the whole difficulty: find the right table,
            # then the right row. Forbidding the overlap there would forbid
            # the only good distractors the kind has.
            if item.kind is not ItemKind.TABLE and gold_text and option in gold_text:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "distractor_overlap",
                        item.id,
                        f"option {letter} sits inside a gold ref — two defensible answers",
                    )
                )
            if item.kind is not ItemKind.TABLE and ref and ref in item.gold_refs:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "distractor_overlap",
                        item.id,
                        f"option {letter} is sourced from a gold ref",
                    )
                )
    return out


def _check_not_stated(
    items: Sequence[Item], text_of: TextOf, readable: Callable[[str], bool]
) -> list[Finding]:
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
        if not readable(item.doc):
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


def _check_quota(items: Sequence[Item], *, require_quota: bool) -> list[Finding]:
    """One item of each kind per paper — the thing that keeps the suite honest
    as it grows.

    Without it, forty papers authored over a week drift toward whatever is
    easiest to write, which is `lookup`, and the benchmark quietly becomes a
    keyword-matching test. With it, every paper owes a table question, a
    cross-reference, something out in an appendix or a caption, and one whose
    answer is not there at all.
    """
    out: list[Finding] = []
    by_doc: dict[str, list[Item]] = {}
    for item in items:
        by_doc.setdefault(item.doc, []).append(item)

    for doc, doc_items in sorted(by_doc.items()):
        kinds = [item.kind for item in doc_items]
        for kind in set(kinds):
            if kinds.count(kind) > 1:
                out.append(
                    Finding(
                        Severity.ERROR,
                        "quota",
                        "",
                        f"{doc}: {kinds.count(kind)} items of kind {kind} — one per paper",
                    )
                )
        missing = [kind for kind in PER_PAPER if kind not in kinds]
        if missing and require_quota:
            out.append(
                Finding(
                    Severity.WARN,
                    "quota",
                    "",
                    f"{doc}: missing {[str(kind) for kind in missing]}",
                )
            )
    return out


def _check_abstract(items: Sequence[Item], abstract_refs: AbstractRefs) -> list[Finding]:
    """A paper carries its own summary at the top, and an item answerable from
    it makes navigation free — the engine reads the first section and stops.

    Refs are checked, not text: paraphrase slips through here, which is exactly
    what the `abstract-only` control is for.
    """
    out: list[Finding] = []
    for item in items:
        inside = set(abstract_refs(item.doc))
        overlap = [ref for ref in item.gold_refs if ref in inside]
        if overlap:
            out.append(
                Finding(
                    Severity.ERROR,
                    "abstract_leak",
                    item.id,
                    f"the answer is in the abstract ({overlap}) — navigation would be free",
                )
            )
    return out


def _check_licences(
    items: Sequence[Item], licenses: Mapping[str, str], policy: LicencePolicy
) -> list[Finding]:
    """What a licence has to permit depends on what the corpus commits.

    Under `redistributable` the corpus carries each paper's text, so anything
    short of CC-BY is an error. Under `fetch-only` it carries hashes, and any
    licence will do — but the licence is still *recorded*, because a corpus
    that cannot say what it is built on is not a corpus, and because switching
    policies later must not require re-reading forty papers.
    """
    out: list[Finding] = []
    for doc in sorted({item.doc for item in items}):
        licence = licenses.get(doc, "")
        if policy is LicencePolicy.FETCH_ONLY:
            if not licence:
                out.append(
                    Finding(
                        Severity.WARN,
                        "licence",
                        "",
                        f"{doc}: no licence recorded — `gbench fetch` fills it from the "
                        f"abstract page; not blocking under {policy}",
                    )
                )
            continue
        if not licence:
            out.append(Finding(Severity.ERROR, "licence", "", f"{doc}: no licence recorded"))
        elif not any(token in licence.lower() for token in REDISTRIBUTABLE):
            out.append(
                Finding(
                    Severity.ERROR,
                    "licence",
                    "",
                    f"{doc}: {licence} does not allow committing the text; either pick a "
                    f"CC-BY paper or set `licence_policy: fetch-only`",
                )
            )
    return out


def failed(findings: Iterable[Finding]) -> bool:
    return any(finding.severity is Severity.ERROR for finding in findings)
