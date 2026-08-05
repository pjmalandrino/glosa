"""What a question is, and how it is put in front of an engine.

Pure data. Nothing here knows that an engine exists, that a document is JSON,
or that anything is scored — `scoring.py` owns the last one.

The one piece of logic that lives here is **rotation**: the same item with its
substantive options cyclically shifted. A small model asked the same question
four times with the answer in four different positions is a position-bias
detector that costs nothing to build, and the rendered prompt is the only
string the bench controls, so it is built in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
"""The substantive options. Four, as MMLU."""

ABSTAIN_LETTER = "E"
"""The sentinel, on every item and never rotated.

It carries "not stated in this document". Present everywhere so that its
appearance is not itself the answer, fixed in place so that "abstain" costs the
same on every item and an engine's abstention signal can map to one letter."""

ABSTAIN_TEXT = "Not stated in this document"

ALL_LETTERS: tuple[str, ...] = (*LETTERS, ABSTAIN_LETTER)

INSTRUCTION = "Answer with a single letter: A, B, C, D or E."
"""The only instruction the bench adds. Byte for byte the same for every engine
— everything an engine does after receiving this string is what is under test."""


class ItemKind(StrEnum):
    LOOKUP = "lookup"
    """The answer sits in one section."""

    CROSSREF = "crossref"
    """It takes two: a definition in one place, applied in another."""

    TABLE = "table"
    """A cell, not a paragraph. Flattening the table loses it."""

    NOT_STATED = "not_stated"
    """The value is absent from the document. `E` is correct."""


@dataclass(frozen=True, slots=True)
class Item:
    """One authored question, with everything needed to score and audit it."""

    id: str
    doc: str
    kind: ItemKind
    question: str
    options: Mapping[str, str]
    """`A`..`D` → option text. `E` is added by `render`, never authored."""
    answer: str
    """`A`..`D`, or `E` on a `not_stated` item."""
    gold_refs: tuple[str, ...] = ()
    """Where the answer lives in the host's projection. The free retrieval
    label on every item — `metrics.hit_at_k` is computed from it."""
    gold_quote: str = ""
    """A literal span of `gold_refs`' text. The linter checks it really is."""
    distractor_refs: Mapping[str, str] = ()  # type: ignore[assignment]
    """Wrong letter → the ref its value was taken from. Distractors come from
    the same document (`EVAL.md` §3.4); this records where, so the linter can
    verify it and a reader can check we did not invent them."""
    removed_from: str = ""
    """`not_stated` only: the ref the value was removed from, so the removal is
    auditable by someone who does not trust us."""
    authored_by: str = ""
    authored_from: str = "document"
    """`document` or `pdf`. Never `trace` — an item written by watching an
    engine read is an item that engine will pass."""

    def __post_init__(self) -> None:
        if self.answer not in ALL_LETTERS:
            raise ValueError(f"{self.id}: answer {self.answer!r} is not one of {ALL_LETTERS}")
        missing = [letter for letter in LETTERS if letter not in self.options]
        if missing:
            raise ValueError(f"{self.id}: missing options {missing}")
        if ABSTAIN_LETTER in self.options:
            raise ValueError(f"{self.id}: option {ABSTAIN_LETTER} is added by the bench")
        if (self.kind is ItemKind.NOT_STATED) != (self.answer == ABSTAIN_LETTER):
            raise ValueError(f"{self.id}: kind {self.kind} and answer {self.answer} disagree")

    def render(self, rotation: int = 0) -> RenderedItem:
        """The item as an engine sees it, with the options shifted by `rotation`.

        Shifting moves option text to a *later* letter: at rotation 1 what was
        `A` is presented as `B`. The gold letter moves with it.
        """
        shift = rotation % len(LETTERS)
        shown: dict[str, str] = {}
        origin: dict[str, str] = {}
        for i, source in enumerate(LETTERS):
            target = LETTERS[(i + shift) % len(LETTERS)]
            shown[target] = self.options[source]
            origin[target] = source
        shown[ABSTAIN_LETTER] = ABSTAIN_TEXT
        origin[ABSTAIN_LETTER] = ABSTAIN_LETTER

        gold = ABSTAIN_LETTER
        if self.answer != ABSTAIN_LETTER:
            gold = LETTERS[(LETTERS.index(self.answer) + shift) % len(LETTERS)]

        body = "\n".join(f"{letter}. {shown[letter]}" for letter in ALL_LETTERS)
        return RenderedItem(
            item=self,
            rotation=shift,
            options=shown,
            origin=origin,
            gold=gold,
            prompt=f"{self.question}\n\n{body}\n\n{INSTRUCTION}",
        )


@dataclass(frozen=True, slots=True)
class RenderedItem:
    """An `Item` at one rotation — the exact string handed to every engine."""

    item: Item
    rotation: int
    options: Mapping[str, str]
    origin: Mapping[str, str]
    """Shown letter → the letter it was authored as. Lets a per-option analysis
    survive rotation."""
    gold: str
    prompt: str

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def abstain_letter(self) -> str:
        return ABSTAIN_LETTER

    @property
    def expects_abstention(self) -> bool:
        return self.item.kind is ItemKind.NOT_STATED


def rotations_of(item: Item, count: int) -> Sequence[RenderedItem]:
    """`count` rotations of `item`, starting at 0.

    Four is the full cycle; one is the smoke profile. Anything between is a
    partial sweep of position bias, which is still worth more than none.
    """
    if not 1 <= count <= len(LETTERS):
        raise ValueError(f"rotations must be 1..{len(LETTERS)}, got {count}")
    return [item.render(index) for index in range(count)]


def unrotate(letter: str, rotation: int) -> str:
    """The letter an option was authored as, given the rotation it was shown at.

    What makes `flip_rate` computable: two rotations agree when they picked the
    same *option*, not the same letter.
    """
    if letter == ABSTAIN_LETTER:
        return ABSTAIN_LETTER
    if letter not in LETTERS:
        raise ValueError(f"{letter!r} is not an option letter")
    return LETTERS[(LETTERS.index(letter) - rotation) % len(LETTERS)]


def with_options(item: Item, options: Mapping[str, str]) -> Item:
    """Copy of `item` with different option text — used by the corpus loader."""
    return replace(item, options=dict(options))
