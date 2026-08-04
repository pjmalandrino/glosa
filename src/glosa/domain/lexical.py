"""Lexical retrieval over a projected document — BM25, no dependency.

The upstream loop spends every hop asking a model to read a table of contents.
That is expensive, it caps out when the outline no longer fits, and it fails
silently on documents where headings are uninformative ("Article 7").

A lexical prior costs microseconds and answers a different question: *which
parts of this document mention what was asked?* It is wrong in a predictable
way — it misses paraphrase — so it never decides alone: it produces a shortlist
that the reading loop confirms, and the loop falls back to model navigation
when the shortlist comes back dry.

Pure domain logic: tokenizing and counting, no I/O.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

K1 = 1.5
"""Term-frequency saturation. The standard 1.2 to 2.0 range; 1.5 is the usual default."""
B = 0.75
"""Length normalization."""
MIN_TOKEN_LEN = 2


def normalize(text: str) -> str:
    """Casefold and strip diacritics.

    Documents and questions rarely agree on accents — "pénalité" in the
    contract, "penalite" in the question typed in a hurry. Stripping them costs
    nothing and removes a whole class of misses on French corpora.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_TOKEN = re.compile(r"[0-9a-z]+(?:[.,][0-9]+)*")
_HAS_DIGIT = re.compile(r"[0-9]")


def fold(token: str) -> str:
    """Collapse the commonest inflections so a query and a document agree.

    Deliberately *not* a linguistic stemmer: no dependency, no language
    detection, no verb morphology. It exists to stop the failures that dominate
    in practice — "penalties" vs "penalty", "livraisons" vs "livraison",
    "travaux" vs "travail" — and it is applied identically to the corpus and to
    the query, so *consistency* matters more than correctness. `traval` is not a
    word; it is a bucket both spellings land in. Tokens holding a digit are
    never touched: `12.4` and `2` are literals.
    """
    if len(token) <= 3 or _HAS_DIGIT.search(token):
        return token

    if token.endswith("aux"):
        # Before the generic -x rule, which would leave "travau".
        token = token[:-3] + "al"
    elif token.endswith("ies"):
        token = token[:-3] + "y"
    elif token.endswith(("ches", "shes", "sses", "xes", "zes")):
        token = token[:-2]
    elif token.endswith("ss"):
        pass
    elif token.endswith("s") or (token.endswith("x") and len(token) > 4):
        token = token[:-1]

    # French -ail/-aux collapse onto -al so travail/travaux and
    # journal/journaux meet at the same stem.
    if token.endswith("ail"):
        token = token[:-3] + "al"
    return token


def tokenize(text: str) -> list[str]:
    """Split into comparable tokens, keeping numbers intact.

    `12.4` and `2` are exactly the tokens a numeric question turns on, so the
    pattern keeps decimal groups together rather than shattering them.
    """
    return [
        fold(t) for t in _TOKEN.findall(normalize(text)) if len(t) >= MIN_TOKEN_LEN or t.isdigit()
    ]


@dataclass(frozen=True, slots=True)
class Scored:
    """One retrieval result."""

    ref: str
    score: float


class Bm25Index:
    """BM25 over a small corpus held in memory.

    "Small" is the point: the corpus is the units of one document, typically
    tens to a few hundred entries. No inverted index, no persistence — building
    it is cheaper than the first token of an LLM call.
    """

    def __init__(self, entries: Iterable[tuple[str, str]]) -> None:
        self._tf: dict[str, dict[str, int]] = {}
        self._len: dict[str, int] = {}
        self._df: dict[str, int] = {}

        for ref, text in entries:
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf[ref] = counts
            self._len[ref] = len(tokens)
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1

        self._n = len(self._tf)
        total = sum(self._len.values())
        self._avgdl = (total / self._n) if self._n else 0.0

    def __len__(self) -> int:
        return self._n

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str, ref: str) -> float:
        counts = self._tf.get(ref)
        if counts is None or self._avgdl == 0:
            return 0.0
        length = self._len[ref]
        total = 0.0
        for token in tokenize(query):
            freq = counts.get(token, 0)
            if not freq:
                continue
            denominator = freq + K1 * (1 - B + B * length / self._avgdl)
            total += self._idf(token) * freq * (K1 + 1) / denominator
        return total

    def rank(self, query: str, *, limit: int | None = None) -> list[Scored]:
        """Refs with a non-zero score, best first.

        Refs that match nothing are omitted rather than returned with a score
        of zero: "no lexical signal" and "ranked last" are different facts, and
        the caller needs to tell them apart to decide whether to trust the
        shortlist at all.
        """
        scored = [Scored(ref, self.score(query, ref)) for ref in self._tf]
        hits = sorted(
            (s for s in scored if s.score > 0.0),
            key=lambda s: (-s.score, s.ref),
        )
        return hits if limit is None else hits[:limit]

    def rare_terms(self, ref: str, *, max_share: float = 0.5) -> frozenset[str]:
        """Terms of `ref` that at most `max_share` of the corpus also uses.

        The cheap test for "does this entry say anything the others do not".
        An opening line repeated across every section of a contract has every
        term at `df = n` and comes back empty; one word of its own is enough to
        come back non-empty.
        """
        counts = self._tf.get(ref)
        if not counts or self._n == 0:
            return frozenset()
        ceiling = max(1, int(self._n * max_share))
        return frozenset(t for t in counts if self._df.get(t, 0) <= ceiling)

    def covered_terms(self, query: str, refs: Sequence[str]) -> frozenset[str]:
        """Query terms that appear in at least one of `refs`."""
        wanted = set(tokenize(query))
        found = {
            token for ref in refs for token in wanted if self._tf.get(ref, {}).get(token, 0) > 0
        }
        return frozenset(found)

    def missing_terms(self, query: str, refs: Sequence[str]) -> frozenset[str]:
        """Query terms nothing in `refs` mentions — a cheap "keep looking" signal."""
        return frozenset(self._answerable(query) - self.covered_terms(query, refs))

    def coverage(self, query: str, refs: Sequence[str]) -> float:
        """Share of the *answerable* query terms that `refs` actually contain.

        Answerable means "present somewhere in this document": a word the
        document never uses says nothing about how well a section matches, so
        counting it would punish every candidate equally and measure nothing.

        Returns 1.0 when the question shares no vocabulary with the document at
        all — there is nothing to be covered, and the caller decides what to do
        with a shortlist built on nothing.
        """
        answerable = self._answerable(query)
        if not answerable:
            return 1.0
        return len(self.covered_terms(query, refs)) / len(answerable)

    def _answerable(self, query: str) -> set[str]:
        return {t for t in tokenize(query) if self._df.get(t, 0) > 0}
