"""The pure half — the half that can silently be wrong.

A scorer, a letter parser and a metric produce numbers whether or not they are
correct, and nobody notices a benchmark that is quietly two points generous.
So the parts with no model in them get tests, and the parts that cannot be
tested — does an 8B read a contract — get a number instead.
"""

from __future__ import annotations

import pytest

from gbench.domain.item import ABSTAIN_LETTER, Item, ItemKind, rotations_of, unrotate
from gbench.domain.metrics import (
    abstention,
    accuracy,
    flip_rate,
    hit_at_k,
    paired_delta,
    percentile,
    read_precision,
)
from gbench.domain.report import DASH, cell
from gbench.domain.scoring import Outcome, Policy, extract_letter, score


def item(**overrides) -> Item:
    base = dict(
        id="q1",
        doc="d",
        kind=ItemKind.LOOKUP,
        question="Rate?",
        options={"A": "one", "B": "two", "C": "three", "D": "four"},
        answer="B",
    )
    base.update(overrides)
    return Item(**base)  # type: ignore[arg-type]


# --- the item and its rotations ------------------------------------------------


def test_rotation_moves_the_option_and_the_gold_together():
    rendered = item().render(1)
    assert rendered.options["B"] == "one"  # what was A is shown as B
    assert rendered.options["C"] == "two"
    assert rendered.gold == "C"  # ...so the answer moved too


def test_the_sentinel_is_on_every_item_and_never_rotates():
    for rotation in range(4):
        rendered = item().render(rotation)
        assert rendered.options[ABSTAIN_LETTER] == "Not stated in this document"
        assert rendered.gold != ABSTAIN_LETTER
    # Otherwise its presence would announce the answer on the not-stated items.
    absent = item(kind=ItemKind.NOT_STATED, answer="E").render(2)
    assert absent.gold == ABSTAIN_LETTER


def test_unrotate_recovers_the_authored_option():
    assert unrotate("C", 1) == "B"
    assert unrotate(ABSTAIN_LETTER, 3) == ABSTAIN_LETTER


def test_a_not_stated_item_must_answer_e():
    with pytest.raises(ValueError, match="disagree"):
        item(kind=ItemKind.NOT_STATED, answer="B")


def test_the_prompt_is_the_same_string_for_everyone():
    prompt = item().render(0).prompt
    assert "A. one" in prompt and "E. Not stated in this document" in prompt
    assert prompt.endswith("Answer with a single letter: A, B, C, D or E.")


def test_rotations_of_refuses_a_partial_cycle_it_cannot_produce():
    assert len(rotations_of(item(), 4)) == 4
    with pytest.raises(ValueError):
        rotations_of(item(), 5)


# --- the letter parser ---------------------------------------------------------

OPTIONS = {"A": "one", "B": "two", "C": "three", "D": "four", "E": "Not stated"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", "B"),
        (" (C) ", "C"),
        ("D. four", "D"),
        ("A) because the clause says so", "A"),
        ("The answer is C.", "C"),
        ("two", "B"),  # exact option text
        ("", None),
        ("I could not find it", None),
        ("Either B or C would fit", None),  # two letters: no coin flip
    ],
)
def test_extract_letter(text, expected):
    assert extract_letter(text, OPTIONS) == expected


def test_a_long_option_quoted_verbatim_counts():
    options = {**OPTIONS, "B": "1/1000 du montant du marché"}
    assert extract_letter("Le taux est de 1/1000 du montant du marché.", options) == "B"


def test_folding_survives_case_and_accents():
    options = {**OPTIONS, "A": "Pénalités de retard mensuelles"}
    assert extract_letter("PENALITES DE RETARD MENSUELLES", options) == "A"


# --- scoring -------------------------------------------------------------------


def test_unparsed_is_wrong_and_named():
    run = score(item().render(0), engine="e", text="hmm", abstained=False)
    assert run.outcome is Outcome.UNPARSED and not run.correct


def test_declared_policy_honours_the_engines_own_signal():
    absent = item(kind=ItemKind.NOT_STATED, answer="E").render(0)
    run = score(absent, engine="glosa", text="I could not find it", abstained=True)
    assert run.chosen == ABSTAIN_LETTER and run.correct


def test_strict_policy_ignores_it_which_is_the_whole_comparison():
    absent = item(kind=ItemKind.NOT_STATED, answer="E").render(0)
    run = score(
        absent, engine="glosa", text="I could not find it", abstained=True, policy=Policy.STRICT
    )
    assert run.outcome is Outcome.UNPARSED


def test_a_contradiction_is_recorded_not_silently_broken():
    run = score(item().render(0), engine="glosa", text="B", abstained=True)
    assert run.chosen == ABSTAIN_LETTER and run.conflicted


def test_an_engine_that_raised_is_an_error_not_a_wrong_answer():
    run = score(item().render(0), engine="e", text="", abstained=False, errored=True)
    assert run.outcome is Outcome.ERROR and run.chosen is None


# --- metrics -------------------------------------------------------------------


def runs(engine: str, results: dict[str, list[bool]]):
    out = []
    for item_id, per_rotation in results.items():
        for rotation, right in enumerate(per_rotation):
            gold = "B"
            out.append(
                score(
                    item(id=item_id).render(0),
                    engine=engine,
                    text=gold if right else "A",
                    abstained=False,
                )
            )
            _ = rotation
    return out


def test_an_item_is_one_observation_not_four():
    # One item answered right 3 times of 4 is 0.75, not three successes.
    scored = runs("e", {"q1": [True, True, True, False], "q2": [False] * 4})
    assert accuracy(scored) == pytest.approx(0.375)


def test_flip_rate_counts_items_that_change_their_pick():
    stable = score(item(id="a").render(0), engine="e", text="B", abstained=False)
    same = score(item(id="a").render(1), engine="e", text="C", abstained=False)
    # rotation 1 shows the authored B as C — so those two agree on the option.
    assert flip_rate([stable, same]) == 0.0
    moved = score(item(id="b").render(1), engine="e", text="A", abstained=False)
    other = score(item(id="b").render(0), engine="e", text="A", abstained=False)
    assert flip_rate([moved, other]) == 1.0


def test_abstention_is_reported_as_a_pair_because_either_alone_is_gameable():
    absent = item(id="n1", kind=ItemKind.NOT_STATED, answer="E").render(0)
    present = item(id="a1").render(0)
    scored = [
        score(absent, engine="e", text="", abstained=True),
        score(present, engine="e", text="", abstained=True),
    ]
    recall, false = abstention(scored, expected={"n1": True, "a1": False})
    assert recall == 1.0 and false == 1.0  # always abstaining: perfect, and useless


def test_hit_at_k_respects_reading_order():
    assert hit_at_k(["#/texts/1", "#/texts/6"], ["#/texts/6"], 3)
    assert not hit_at_k(["#/texts/1", "#/texts/6"], ["#/texts/6"], 1)
    assert not hit_at_k([], ["#/texts/6"], 3)


def test_read_precision_is_none_when_nothing_was_reported():
    assert read_precision([], ["#/a"], {"#/a": 10}) is None
    assert read_precision(["#/a", "#/b"], ["#/a"], {"#/a": 10, "#/b": 30}) == 0.25


def test_percentile_is_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.5) is None


def test_paired_delta_says_not_separable_when_it_is_not():
    a = runs("a", {f"q{i}": [i % 2 == 0] * 4 for i in range(20)})
    b = runs("b", {f"q{i}": [i % 2 == 0] * 4 for i in range(20)})
    interval = paired_delta(a, b)
    assert interval.point == 0.0 and not interval.separable


def test_paired_delta_finds_a_real_difference():
    a = runs("a", {f"q{i}": [True] * 4 for i in range(20)})
    b = runs("b", {f"q{i}": [False] * 4 for i in range(20)})
    interval = paired_delta(a, b)
    assert interval.point == pytest.approx(100.0) and interval.separable


# --- report --------------------------------------------------------------------


def test_a_metric_an_engine_cannot_report_is_a_dash_not_a_zero():
    assert cell(None) == DASH
    assert cell(0.0, pct=True) == "0.0 %"
