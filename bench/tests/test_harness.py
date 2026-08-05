"""The harness end to end, on a scripted model.

No GPU, no network, no competitor installed — B0's acceptance bar (`EVAL.md`
§7): the whole path from corpus to table runs in CI, so a broken adapter or a
mis-shaped journal row is caught by `pytest` rather than by an overnight run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from glosa.ports.chat import ChatModel, Message
from pydantic import BaseModel

from gbench.domain.item import ItemKind, assigned_rotation
from gbench.domain.lint import Severity, failed, lint
from gbench.domain.scoring import Policy
from gbench.infra.corpus import FileCorpus
from gbench.infra.engines.controls import (
    AbstractOnlyEngine,
    ClosedBookEngine,
    OracleContextEngine,
)
from gbench.infra.journal import Journal, config_hash
from gbench.runner import RunPlan, run
from gbench.scoreboard import build

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


class ScriptedModel:
    """Replies with whatever it is told to, in order. Implements glosa's port."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "scripted"

    def for_model(self, model_id: str) -> ChatModel:
        return self  # type: ignore[return-value]

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        self.prompts.append(messages[-1].content)
        return self._replies[min(len(self.prompts) - 1, len(self._replies) - 1)]

    async def structured[T: BaseModel](
        self, messages: list[Message], *, schema: type[T], max_tokens: int | None = None
    ) -> T:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


@pytest.fixture
def corpus() -> FileCorpus:
    return FileCorpus(CORPUS)


def lint_all(corpus, items=None):
    return lint(
        corpus.items() if items is None else items,
        text_of=corpus.text_of,
        doc_text=corpus.doc_text,
        abstract_refs=corpus.abstract_refs,
        licenses=corpus.licenses(),
    )


def test_the_fixture_corpus_passes_its_own_linter(corpus):
    findings = lint_all(corpus)
    assert not findings, [str(f) for f in findings]


def test_every_paper_owes_one_item_of_each_kind(corpus):
    kinds = [item.kind for item in corpus.items("toy-ccap")]
    assert sorted(kinds) == sorted(ItemKind)


def test_sections_json_has_not_drifted_from_the_conversion(corpus):
    # It is generated from docling.json and committed separately, so it can
    # rot. The linter checks quotes against it; if it disagrees with the
    # conversion the engines read, every gold_ref is a lie.
    for slug in corpus.slugs:
        if corpus.converted(slug):
            assert corpus.sections(slug) == corpus.project(slug), slug


def test_the_abstract_is_the_front_matter_and_not_the_whole_paper(corpus):
    # A contract has no abstract — only its title precedes the first heading.
    # The rule that produces that is the same one that stops at "1 Introduction"
    # on a paper, and getting it wrong flags every item as an abstract leak.
    assert corpus.abstract_refs("toy-ccap") == ("#/texts/0",)


def test_the_linter_catches_a_quote_that_is_not_in_the_document(corpus):
    from dataclasses import replace

    tampered = [replace(corpus.items()[0], gold_quote="pénalités fixées à 3 % du montant")]
    findings = lint_all(corpus, tampered)
    assert failed(findings)
    assert any(f.check == "gold_quote" for f in findings if f.severity is Severity.ERROR)


def test_the_linter_catches_an_item_authored_from_a_trace(corpus):
    from dataclasses import replace

    tampered = [replace(corpus.items()[0], authored_from="trace")]
    assert failed(lint_all(corpus, tampered))


def test_the_linter_catches_a_distractor_whose_source_is_a_lie(corpus):
    from dataclasses import replace

    item = next(i for i in corpus.items() if i.id == "toy-q01")
    tampered = [replace(item, distractor_refs={**item.distractor_refs, "A": "#/texts/6"})]
    findings = lint_all(corpus, tampered)
    assert failed(findings)
    assert any(f.check == "distractor_source" for f in findings)


def test_the_linter_catches_an_item_answerable_from_the_abstract(corpus):
    from dataclasses import replace

    tampered = [replace(corpus.items()[0], gold_refs=("#/texts/0",))]
    findings = lint_all(corpus, tampered)
    assert any(f.check == "abstract_leak" for f in findings if f.severity is Severity.ERROR)


def test_the_linter_refuses_a_paper_we_cannot_redistribute(corpus):
    findings = lint(
        corpus.items(),
        text_of=corpus.text_of,
        doc_text=corpus.doc_text,
        licenses={"toy-ccap": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"},
    )
    assert failed(findings)
    assert any(f.check == "licence" for f in findings)


def test_a_table_item_may_take_its_distractors_from_its_own_table(corpus):
    # The other rows of the same table are the only good distractors a table
    # question has, and finding the right table then the wrong row is exactly
    # the failure the kind exists to catch.
    table_item = next(i for i in corpus.items() if i.kind is ItemKind.TABLE)
    assert "#/tables/0" in table_item.gold_refs
    assert not failed(lint_all(corpus, [table_item]))


def test_gold_refs_resolve_through_glosas_own_projection(corpus):
    # If they did not, no engine could ever hit them and `hit@k` would be a
    # measurement of the corpus rather than of the engines.
    for item in corpus.items():
        for ref in item.gold_refs:
            assert corpus.text_of(item.doc, ref), f"{item.id}: {ref} resolves to nothing"


async def test_run_then_score_with_no_model_of_any_kind(tmp_path, corpus):
    items = list(corpus.items())
    documents = {"toy-ccap": corpus.document("toy-ccap")}
    journal = Journal(tmp_path / "journal.jsonl")

    # A model that always answers "B": right on q01 at rotation 0, wrong
    # elsewhere — enough to prove the arithmetic flows through.
    engines = [
        ClosedBookEngine(ScriptedModel(["B"])),
        AbstractOnlyEngine(ScriptedModel(["B"]), corpus.abstract_text),
        OracleContextEngine(ScriptedModel(["B"]), corpus.text_of),
    ]
    rows = await run(
        engines=engines,
        documents=documents,
        items=items,
        journal=journal,
        config=config_hash({"model": "scripted"}),
        plan=RunPlan(sweep=True, concurrency=1),
    )
    assert len(rows) == len(items) * 4 * len(engines)

    board = build(journal.read(), items, policy=Policy.DECLARED, exclude_leaky=False)
    names = {row.engine for row in board.rows}
    assert names == {"closed-book", "abstract-only", "oracle-context"}
    # MMLU reports per subject; this reports per kind, and every kind is there.
    assert set(board.by_kind["closed-book"]) == {str(kind) for kind in ItemKind}
    for row in board.rows:
        assert 0.0 <= row.accuracy <= 1.0
        # Neither control has an abstention signal, so `E` never comes for free.
        assert row.abstention_recall == 0.0


async def test_the_oracle_is_handed_the_gold_section_and_nothing_else(corpus):
    model = ScriptedModel(["B"])
    engine = OracleContextEngine(model, corpus.text_of)
    item = next(i for i in corpus.items() if i.id == "toy-q01")
    answer = await engine.answer(corpus.document("toy-ccap"), item.render(0))
    assert answer.read_refs == ("#/texts/8",)
    assert "1/1000" in model.prompts[0]
    assert "Retenue de garantie" not in model.prompts[0]


async def test_the_abstract_control_sees_the_front_matter_and_nothing_else(corpus):
    model = ScriptedModel(["B"])
    engine = AbstractOnlyEngine(model, corpus.abstract_text)
    item = next(i for i in corpus.items() if i.id == "toy-q01")
    await engine.answer(corpus.document("toy-ccap"), item.render(0))
    # The title is front matter; the article that answers the question is not.
    # (The rate itself appears in the prompt — it is one of the options.)
    assert "Marché public de travaux" in model.prompts[0]
    assert "Article 7" not in model.prompts[0]


async def test_a_journal_row_survives_a_round_trip(tmp_path, corpus):
    journal = Journal(tmp_path / "j.jsonl")
    await run(
        engines=[ClosedBookEngine(ScriptedModel(["C"]))],
        documents={"toy-ccap": corpus.document("toy-ccap")},
        items=list(corpus.items())[:1],
        journal=journal,
        config="cfg",
        plan=RunPlan(flip_sample=0, concurrency=1),
    )
    (row,) = list(journal.read())
    assert row.engine == "closed-book" and row.text == "C"
    assert row.options["E"] == "Not stated in this document"


async def test_leaky_items_are_flagged_and_dropped_from_the_headline(tmp_path, corpus):
    """An item a no-navigation control answers was never about the document."""
    items = [i for i in corpus.items() if i.id == "toy-q01"]
    journal = Journal(tmp_path / "j.jsonl")
    # Unswept items run at their assigned rotation, not at rotation 0 — so the
    # letter that is correct without the document has to be computed, which is
    # itself worth pinning: a test that hardcoded "B" would pass only while
    # `assigned_rotation` happened to return 0.
    gold = items[0].render(assigned_rotation(items[0].id)).gold
    await run(
        engines=[ClosedBookEngine(ScriptedModel([gold]))],  # right, with no document
        documents={"toy-ccap": corpus.document("toy-ccap")},
        items=items,
        journal=journal,
        config="cfg",
        plan=RunPlan(flip_sample=0, concurrency=1),
    )
    board = build(journal.read(), items)
    assert board.leaky == ("toy-q01",)
