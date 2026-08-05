"""Answer text → a letter → an outcome.

Two rules govern this module, and both are there to stop the scorer becoming a
second experiment.

**One parser, no repair.** Every engine's answer goes through the same
deterministic extractor. There is no repair prompt and no judge model: an
engine that cannot emit a letter scores zero and gets a named metric for it
(`unparsed_rate`), because that is exactly the `find_json_dicts(...)[0]` failure
class of `DESIGN.md` §4.1 expressed as a percentage.

**Two policies over one journal.** `DECLARED` honours the engine's own
abstention signal; `STRICT` counts only a literal `E` in the text. Scoring runs
offline (`EVAL.md` §4.3), so both numbers come from the same execution — and
the gap between them is what a typed `RunStatus` is worth.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from gbench.domain.item import ABSTAIN_LETTER, ALL_LETTERS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gbench.domain.item import RenderedItem

_ONLY_LETTER = re.compile(r"^\W*([A-E])\W*$")
_LEADING_LETTER = re.compile(r"^\W*\(?([A-E])\)?\s*[).:\--—]\s")
_STANDALONE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


class Outcome(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSED = "unparsed"
    """No letter recoverable. Counts as wrong, reported on its own line."""
    ERROR = "error"
    """The engine raised or timed out. Also wrong, also reported separately —
    an engine that crashes on 10 % of items has not scored 0.9 x its accuracy,
    it has a bug worth naming."""


class Policy(StrEnum):
    DECLARED = "declared"
    """The engine's declared abstention signal maps to `E`."""

    STRICT = "strict"
    """Only a letter in the answer text counts."""


def fold(text: str) -> str:
    """Case, accents and whitespace folded — the same folding glosa's citation
    check uses, so "1/1000" matches across a soft hyphen and "PÉNALITÉS"
    matches "penalites"."""
    stripped = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def extract_letter(text: str, options: Mapping[str, str]) -> str | None:
    """The one letter this answer commits to, or `None`.

    Tried in order, most explicit first. Each rule must be unambiguous: a rule
    that matches two different letters yields nothing rather than a coin flip.
    """
    if not text or not text.strip():
        return None
    body = text.strip()

    only = _ONLY_LETTER.match(body)
    if only:
        return only.group(1)

    leading = _LEADING_LETTER.match(body)
    if leading:
        return leading.group(1)

    folded = fold(body)
    exact = [letter for letter, option in options.items() if fold(option) == folded]
    if len(exact) == 1:
        return exact[0]

    letters = {match.group(1) for match in _STANDALONE.finditer(body)}
    if len(letters) == 1:
        return letters.pop()

    quoted = [
        letter
        for letter, option in options.items()
        if letter != ABSTAIN_LETTER and len(fold(option)) >= 8 and fold(option) in folded
    ]
    if len(quoted) == 1:
        return quoted[0]

    return None


@dataclass(frozen=True, slots=True)
class ScoredRun:
    """One (item, engine, rotation) scored under one policy."""

    item_id: str
    engine: str
    rotation: int
    policy: Policy
    outcome: Outcome
    chosen: str | None
    gold: str
    abstained: bool
    conflicted: bool = False
    """The engine declared abstention while its text named a different letter.

    Not scored — recorded. An engine that says `not_in_document` and then
    quotes a value is contradicting itself, and that is worth a number rather
    than a silent tie-break."""

    @property
    def correct(self) -> bool:
        return self.outcome is Outcome.CORRECT


def score(
    rendered: RenderedItem,
    *,
    engine: str,
    text: str,
    abstained: bool,
    errored: bool = False,
    policy: Policy = Policy.DECLARED,
) -> ScoredRun:
    """Score one answer. Never sees a model, never makes a call."""
    from_text = extract_letter(text, rendered.options)

    chosen = from_text
    conflicted = False
    if policy is Policy.DECLARED and abstained:
        # The typed signal wins over the prose. An engine reporting
        # `not_in_document` has answered the question the option list asks.
        conflicted = from_text is not None and from_text != ABSTAIN_LETTER
        chosen = ABSTAIN_LETTER

    if errored:
        outcome = Outcome.ERROR
    elif chosen is None:
        outcome = Outcome.UNPARSED
    elif chosen == rendered.gold:
        outcome = Outcome.CORRECT
    else:
        outcome = Outcome.INCORRECT

    return ScoredRun(
        item_id=rendered.id,
        engine=engine,
        rotation=rendered.rotation,
        policy=policy,
        outcome=outcome,
        chosen=chosen if outcome is not Outcome.ERROR else None,
        gold=rendered.gold,
        abstained=abstained,
        conflicted=conflicted,
    )


def letter_distribution(golds: list[str]) -> dict[str, int]:
    """How often each letter is the answer. Feeds the linter's balance check."""
    counts = dict.fromkeys(ALL_LETTERS, 0)
    for gold in golds:
        counts[gold] = counts.get(gold, 0) + 1
    return counts
