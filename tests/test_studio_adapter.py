"""The Studio-facing runner: legacy shape, host factories, caching, errors."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from docling_core.types.doc import DoclingDocument

from glosa.adapters.studio import GlosaReasoningRunner
from glosa.document.index import DocIndex
from glosa.errors import ReasoningParseError
from glosa.strategy.navigate import NavigateConfig, Reading, Selection
from glosa.types import LegacyIteration, RunStatus, Step, Trace
from tests.conftest import FakeChatModel

LOOP = NavigateConfig(direct_char_threshold=0)


# --- Studio's domain types, transcribed (see contract/test_studio_port.py) ---


@dataclass(frozen=True)
class StudioIteration:
    iteration: int
    section_ref: str
    reason: str
    section_text_length: int
    can_answer: bool
    response: str


@dataclass(frozen=True)
class StudioResult:
    answer: str
    iterations: list[StudioIteration]
    converged: bool


class StudioParseError(Exception):
    def __init__(self, model_id: str, reason: str = "no parseable answer") -> None:
        super().__init__(f"{model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


# --- legacy shape ------------------------------------------------------------


async def test_iterations_expose_exactly_the_six_legacy_fields(
    flat_doc: DoclingDocument,
) -> None:
    """Studio does `ReasoningIteration(**it.model_dump())`. An extra key there
    is a TypeError at runtime, so the projection must stay exactly this wide."""
    runner = GlosaReasoningRunner(FakeChatModel([Reading(sufficient=True, response="12.4M")]))
    result = await runner.run(document_json=flat_doc.model_dump_json(), query="revenue?")

    assert isinstance(result.iterations[0], LegacyIteration)
    assert set(result.iterations[0].model_dump()) == {
        "iteration",
        "section_ref",
        "reason",
        "section_text_length",
        "can_answer",
        "response",
    }
    assert result.converged is True


async def test_host_factories_produce_the_hosts_own_types(flat_doc: DoclingDocument) -> None:
    """This is what removes the translation adapter on Studio's side."""
    runner = GlosaReasoningRunner(
        FakeChatModel([Reading(sufficient=True, response="12.4M")]),
        result_factory=StudioResult,
        iteration_factory=StudioIteration,
    )
    result = await runner.run(document_json=flat_doc.model_dump_json(), query="revenue?")

    assert isinstance(result, StudioResult)
    assert isinstance(result.iterations[0], StudioIteration)
    assert result.iterations[0].section_ref == "#/body"


async def test_a_non_answer_is_labelled_in_the_answer_text(flat_doc: DoclingDocument) -> None:
    """A UI that renders only `answer` must still learn the run failed."""
    runner = GlosaReasoningRunner(
        FakeChatModel([Reading(sufficient=False, response="Not covered.", absent=True)])
    )
    result = await runner.run(document_json=flat_doc.model_dump_json(), query="cats?")

    assert result.converged is False
    assert result.answer.startswith("[not answered")
    assert "Not covered." in result.answer


async def test_status_annotation_can_be_turned_off(flat_doc: DoclingDocument) -> None:
    runner = GlosaReasoningRunner(
        FakeChatModel([Reading(sufficient=False, response="Not covered.", absent=True)]),
        annotate_status=False,
    )
    result = await runner.run(document_json=flat_doc.model_dump_json(), query="cats?")
    assert result.answer == "Not covered."


# --- errors ------------------------------------------------------------------


async def test_parse_failures_are_raised_as_the_hosts_exception(
    flat_doc: DoclingDocument,
) -> None:
    """Studio maps its own `ReasoningParseError` to a 502 with guidance."""
    runner = GlosaReasoningRunner(
        FakeChatModel([ReasoningParseError(model_id="granite3.3:8b", reason="nope")]),
        parse_error_factory=StudioParseError,
    )
    with pytest.raises(StudioParseError) as excinfo:
        await runner.run(document_json=flat_doc.model_dump_json(), query="q")

    assert excinfo.value.model_id == "granite3.3:8b"
    assert excinfo.value.reason == "nope"


# --- caching and overrides ---------------------------------------------------


async def test_the_document_is_parsed_once_across_queries(
    flat_doc: DoclingDocument, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Studio asks several questions of the same stored analysis."""
    calls = {"n": 0}
    original = DocIndex.from_json

    def counted(document_json: str) -> DocIndex:
        calls["n"] += 1
        return original(document_json)

    monkeypatch.setattr(DocIndex, "from_json", staticmethod(counted))

    payload = flat_doc.model_dump_json()
    runner = GlosaReasoningRunner(
        FakeChatModel(
            [
                Reading(sufficient=True, response="a"),
                Reading(sufficient=True, response="b"),
            ]
        )
    )
    await runner.run(document_json=payload, query="one")
    await runner.run(document_json=payload, query="two")

    assert calls["n"] == 1


async def test_the_cache_is_bounded(flat_doc: DoclingDocument, nested_doc: DoclingDocument) -> None:
    runner = GlosaReasoningRunner(
        FakeChatModel(
            [
                Reading(sufficient=True, response="a"),
                Reading(sufficient=True, response="b"),
            ]
        ),
        cache_size=1,
    )
    await runner.run(document_json=flat_doc.model_dump_json(), query="one")
    await runner.run(document_json=nested_doc.model_dump_json(), query="two")
    assert len(runner._cache) == 1


async def test_model_id_override_is_honoured(flat_doc: DoclingDocument) -> None:
    model = FakeChatModel([Reading(sufficient=True, response="ok")])
    runner = GlosaReasoningRunner(model)
    trace = await runner.run_trace(
        document_json=flat_doc.model_dump_json(), query="q", model_id="mistral-small3.2"
    )
    assert trace.model_id == "mistral-small3.2"


async def test_run_trace_exposes_provenance_the_legacy_shape_drops(
    flat_doc: DoclingDocument,
) -> None:
    runner = GlosaReasoningRunner(
        FakeChatModel(
            [
                Selection(reason="here", ref=DocIndex(flat_doc).units[1].ref),
                Reading(sufficient=True, response="12.4M"),
            ]
        ),
        config=LOOP,
    )
    trace = await runner.run_trace(document_json=flat_doc.model_dump_json(), query="revenue?")

    assert isinstance(trace, Trace)
    assert trace.status is RunStatus.ANSWERED
    step: Step = trace.steps[0]
    assert step.pages == (1,)
    assert any(s.bbox is not None for s in step.spans)


async def test_aclose_releases_the_backend(flat_doc: DoclingDocument) -> None:
    model = FakeChatModel([])
    await GlosaReasoningRunner(model).aclose()
    assert model.closed is True
