"""Conformance with Docling Studio's `ReasoningRunner` port.

The protocol below is transcribed from `document-parser/domain/ports.py` in
scub-france/Docling-Studio. It is duplicated here on purpose: glosa must satisfy
the contract **without depending on Studio**, and this test is what stops a
refactor from silently breaking the integration.

Keep it in sync with upstream by transcription, never by import.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from glosa.adapters.studio import GlosaReasoningRunner
from glosa.domain.reading import Reading
from tests.conftest import FakeChatModel


@runtime_checkable
class ReasoningRunner(Protocol):
    """Transcribed from Docling Studio, `domain/ports.py`."""

    @property
    def is_available(self) -> bool: ...

    async def run(
        self,
        *,
        document_json: str,
        query: str,
        model_id: str | None = None,
    ) -> Any: ...


def test_the_runner_satisfies_studios_protocol() -> None:
    runner = GlosaReasoningRunner(FakeChatModel([]))
    assert isinstance(runner, ReasoningRunner)


def test_is_available_does_no_io() -> None:
    """Studio calls this on the request path to choose between serving and 503.

    A model whose backend is down still reports available; `health()` is the
    probe that touches the network.
    """
    assert GlosaReasoningRunner(FakeChatModel([])).is_available is True


async def test_run_accepts_the_ports_keyword_signature(flat_json: str) -> None:
    runner = GlosaReasoningRunner(FakeChatModel([Reading(sufficient=True, response="yes")]))
    result = await runner.run(document_json=flat_json, query="revenue?", model_id=None)
    assert result.answer == "yes"
