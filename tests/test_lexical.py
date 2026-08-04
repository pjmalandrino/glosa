"""Lexical retrieval: tokenizing, BM25, term coverage."""

from __future__ import annotations

import pytest

from glosa.domain.lexical import Bm25Index, fold, normalize, tokenize

CORPUS = [
    ("scope", "The supplier shall deliver the works described in Annex A."),
    ("penalties", "Late delivery incurs a penalty of 2% per week, capped at 10%."),
    ("payment", "Invoices are payable within 30 days of receipt."),
]


@pytest.fixture
def index() -> Bm25Index:
    return Bm25Index(CORPUS)


def test_normalization_folds_case_and_accents() -> None:
    """French corpora and hurried questions rarely agree on accents."""
    assert normalize("Pénalité") == normalize("penalite") == "penalite"


def test_numbers_survive_tokenization() -> None:
    """`12.4` and `30` are exactly what a numeric question turns on."""
    assert "12.4" in tokenize("Revenue reached 12.4M EUR")
    assert "30" in tokenize("payable within 30 days")


def test_single_letters_are_dropped_but_digits_are_not() -> None:
    assert tokenize("a b 7 to") == ["7", "to"]


def test_the_relevant_entry_ranks_first(index: Bm25Index) -> None:
    assert index.rank("what is the late delivery penalty")[0].ref == "penalties"


def test_a_query_matching_nothing_returns_no_hits_at_all(index: Bm25Index) -> None:
    """Empty means "no signal", not "everything scored badly" — the caller
    needs that distinction to decide whether to trust the shortlist."""
    assert index.rank("cryptocurrency custody arrangements") == []


def test_a_term_in_every_unit_carries_almost_no_weight() -> None:
    """IDF is what keeps boilerplate from dominating a shortlist.

    Note the corpus has to be built for this: on three entries, even "the" is
    a rare term, and BM25 is right to say so. Here "delivery" is in all three.
    """
    index = Bm25Index(
        [
            ("scope", "Delivery of the works described in Annex A."),
            ("penalties", "Late delivery incurs a penalty of 2% per week."),
            ("payment", "Delivery of invoices triggers payment within 30 days."),
        ]
    )
    assert index.rank("delivery invoices")[0].ref == "payment"


def test_limit_truncates_the_ranking(index: Bm25Index) -> None:
    assert len(index.rank("delivery the", limit=1)) == 1


def test_missing_terms_flags_what_has_not_been_seen(index: Bm25Index) -> None:
    """Returns *index* tokens — folded — not the spelling the user typed."""
    missing = index.missing_terms("penalty invoices", ["penalties"])
    assert missing == frozenset({"invoice"})


def test_missing_terms_ignores_words_absent_from_the_whole_corpus(index: Bm25Index) -> None:
    """A term nothing mentions is not evidence that we have looked too little."""
    assert index.missing_terms("penalty zebra", ["penalties"]) == frozenset()


def test_an_empty_corpus_scores_nothing() -> None:
    empty = Bm25Index([])
    assert len(empty) == 0
    assert empty.rank("anything") == []


@pytest.mark.parametrize(
    ("inflected", "base"),
    [
        ("penalties", "penalty"),
        ("livraisons", "livraison"),
        ("travaux", "travail"),
        ("journaux", "journal"),
        ("clauses", "clause"),
        ("boxes", "box"),
    ],
)
def test_inflections_collapse_onto_one_bucket(inflected: str, base: str) -> None:
    """Not a linguistic stemmer — a bucket both spellings land in.

    `travail` and `travaux` both fold to `traval`, which is not a word. It does
    not need to be: the same transform runs on the corpus and on the query.
    """
    assert fold(inflected) == fold(base)


@pytest.mark.parametrize("token", ["process", "gas", "12.4", "2", "iso"])
def test_folding_leaves_these_alone(token: str) -> None:
    """Double-s words, short words and anything numeric are literals."""
    assert fold(token) == token


def test_a_plural_query_finds_a_singular_document() -> None:
    """The failure this exists to remove."""
    index = Bm25Index([("clause", "Late delivery incurs a penalty of 2% per week.")])
    assert index.rank("penalties")[0].ref == "clause"
