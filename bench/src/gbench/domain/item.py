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

import hashlib
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
    """Five kinds, and every paper contributes exactly one of each.

    Not a taxonomy for its own sake: each kind is a place where the three
    engines are built differently, so the per-kind breakdown is where the
    headline accuracy stops being one number and starts being an explanation.
    Five kinds x forty papers is also what makes the quota exact — the suite
    cannot drift toward the easy kind, because there is no room for it to.
    """

    LOOKUP = "lookup"
    """A stated fact, in one section of running prose. The baseline."""

    TABLE = "table"
    """A cell of a results table.

    glosa serializes tables as HTML, `docling-agent` flattens them outside page
    mode, PageIndex sees whatever the markdown export produced. Reading the
    right table and the wrong row scores zero, which is the point."""

    CROSSREF = "crossref"
    """Two sections: a symbol defined in the method, used in the results."""

    PERIPHERAL = "peripheral"
    """The answer is outside the main prose flow — a figure caption, a
    footnote, an appendix.

    A reader that walks headings from the top and stops when it has enough
    never gets there. On a paper this is the most common real failure."""

    NOT_STATED = "not_stated"
    """The value is absent from the paper. `E` is correct."""


PER_PAPER: tuple[ItemKind, ...] = tuple(ItemKind)
"""The quota: one item of each kind per paper, checked by the linter."""


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
    domain: str = ""
    """The paper's field — arXiv's primary category, e.g. `cs.CL`, `q-bio.NC`.

    MMLU reports per subject, and so does this: an engine strong on machine
    learning papers and weak on condensed matter has told you something a
    single average hides. Carried on the item rather than looked up at scoring
    time, so a journal stays scoreable on its own."""
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


def assigned_rotation(item_id: str) -> int:
    """The one rotation this item is shown at when the suite is not swept.

    Stable across runs and uniform over a suite of any size, so 200 items put
    the right answer in all four positions about equally often. That kills the
    *aggregate* position bias — which is what would otherwise inflate or deflate
    an engine's headline — at 1x the cost instead of 4x.

    What it cannot measure is per-item instability: whether a given engine
    changes its mind when the options move. That needs the same item at several
    rotations, and `schedule` buys it on a subsample.
    """
    digest = hashlib.sha256(item_id.encode()).digest()
    return digest[0] % len(LETTERS)


def flip_subsample(items: Sequence[Item], size: int) -> tuple[str, ...]:
    """`size` item ids, spread evenly across kinds and papers.

    Deterministic: sorting by (kind, id) and striding gives a sample balanced
    over kinds by construction, and the same sample on every run — so a
    `flip_rate` moving between two runs is the engine moving, not the sample.
    """
    if size <= 0 or not items:
        return ()
    ordered = sorted(items, key=lambda item: (str(item.kind), item.id))
    if size >= len(ordered):
        return tuple(item.id for item in ordered)
    stride = len(ordered) / size
    return tuple(ordered[int(index * stride)].id for index in range(size))


def schedule(
    items: Sequence[Item], *, sweep: bool = False, flip_sample: int = 0
) -> list[RenderedItem]:
    """Every (item, rotation) the run will execute.

    `sweep` runs the full four rotations on everything — right for a small
    suite, wasteful at 200 items across six engines. The default gives each item
    its assigned rotation and sweeps only `flip_sample` of them: 200 items with
    a 40-item sweep is 320 executions instead of 800, and still reports both
    numbers rotation was there to produce.
    """
    if sweep:
        return [rendered for item in items for rendered in rotations_of(item, len(LETTERS))]

    swept = set(flip_subsample(items, flip_sample))
    out: list[RenderedItem] = []
    for item in items:
        if item.id in swept:
            out.extend(rotations_of(item, len(LETTERS)))
        else:
            out.append(item.render(assigned_rotation(item.id)))
    return out


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
