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

from gbench.domain.lint import Severity, failed, lint
from gbench.domain.scoring import Policy
from gbench.infra.corpus import FileCorpus
from gbench.infra.engines.controls import ClosedBookEngine, OracleContextEngine
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


def test_the_fixture_corpus_passes_its_own_linter(corpus):
    findings = lint(corpus.items(), text_of=corpus.text_of, doc_text=corpus.doc_text)
    assert not failed(findings), [str(f) for f in findings]


def test_the_linter_catches_a_quote_that_is_not_in_the_document(corpus):
    from dataclasses import replace

    tampered = [replace(corpus.items()[0], gold_quote="pénalités fixées à 3 % du montant")]
    findings = lint(tampered, text_of=corpus.text_of, doc_text=corpus.doc_text)
    assert failed(findings)
    assert any(f.check == "gold_quote" for f in findings if f.severity is Severity.ERROR)


def test_the_linter_catches_an_item_authored_from_a_trace(corpus):
    from dataclasses import replace

    tampered = [replace(corpus.items()[0], authored_from="trace")]
    findings = lint(tampered, text_of=corpus.text_of, doc_text=corpus.doc_text)
    assert failed(findings)


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
        OracleContextEngine(ScriptedModel(["B"]), corpus.text_of),
    ]
    rows = await run(
        engines=engines,
        documents=documents,
        items=items,
        journal=journal,
        config=config_hash({"model": "scripted"}),
        plan=RunPlan(rotations=2, concurrency=1),
    )
    assert len(rows) == len(items) * 2 * len(engines)

    board = build(journal.read(), items, policy=Policy.DECLARED, exclude_leaky=False)
    names = {row.engine for row in board.rows}
    assert names == {"closed-book", "oracle-context"}
    for row in board.rows:
        assert 0.0 <= row.accuracy <= 1.0
        # Neither control has an abstention signal, so `E` never comes for free.
        assert row.abstention_recall == 0.0


async def test_the_oracle_is_handed_the_gold_section_and_nothing_else(corpus):
    model = ScriptedModel(["B"])
    engine = OracleContextEngine(model, corpus.text_of)
    item = next(i for i in corpus.items() if i.id == "toy-q01")
    answer = await engine.answer(corpus.document("toy-ccap"), item.render(0))
    assert answer.read_refs == ("#/texts/6",)
    assert "1/1000" in model.prompts[0]
    assert "Retenue de garantie" not in model.prompts[0]


async def test_a_journal_row_survives_a_round_trip(tmp_path, corpus):
    journal = Journal(tmp_path / "j.jsonl")
    await run(
        engines=[ClosedBookEngine(ScriptedModel(["C"]))],
        documents={"toy-ccap": corpus.document("toy-ccap")},
        items=list(corpus.items())[:1],
        journal=journal,
        config="cfg",
        plan=RunPlan(rotations=1, concurrency=1),
    )
    (row,) = list(journal.read())
    assert row.engine == "closed-book" and row.text == "C"
    assert row.options["E"] == "Not stated in this document"


async def test_leaky_items_are_flagged_and_dropped_from_the_headline(tmp_path, corpus):
    """An item the closed-book control answers was never about the document."""
    items = [i for i in corpus.items() if i.id == "toy-q01"]
    journal = Journal(tmp_path / "j.jsonl")
    await run(
        engines=[ClosedBookEngine(ScriptedModel(["B"]))],  # right, with no document
        documents={"toy-ccap": corpus.document("toy-ccap")},
        items=items,
        journal=journal,
        config="cfg",
        plan=RunPlan(rotations=1, concurrency=1),
    )
    board = build(journal.read(), items)
    assert board.leaky == ("toy-q01",)
