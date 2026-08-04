"""Shared fixtures: hand-built documents and a scripted chat model."""

from __future__ import annotations

from typing import Any

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ProvenanceItem
from pydantic import BaseModel

from glosa.llm.port import ChatModel, Message

PAGE_HEIGHT = 792.0
PAGE_WIDTH = 612.0


def prov(page_no: int, top: float = 700.0, length: int = 10) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=72.0, t=top, r=540.0, b=top - 14.0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, length),
    )


def _pages(doc: DoclingDocument, count: int) -> None:
    for page_no in range(1, count + 1):
        doc.add_page(page_no=page_no, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))


@pytest.fixture
def nested_doc() -> DoclingDocument:
    """Headings carry their content as children — the well-formed case."""
    doc = DoclingDocument(name="annual-report")
    _pages(doc, 2)
    title = doc.add_title(text="Annual Report 2025", prov=prov(1, 740))
    revenue = doc.add_heading(text="Revenue", level=1, parent=title, prov=prov(1, 700))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Revenue reached 12.4M EUR, up 8% year on year.",
        parent=revenue,
        prov=prov(1, 680),
    )
    risks = doc.add_heading(text="Risks", level=1, parent=title, prov=prov(2, 700))
    legal = doc.add_heading(text="Legal", level=2, parent=risks, prov=prov(2, 660))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="A supplier dispute is pending before the commercial court.",
        parent=legal,
        prov=prov(2, 640),
    )
    return doc


@pytest.fixture
def flat_doc() -> DoclingDocument:
    """Same content, every item a sibling of the body — what most PDFs produce.

    Upstream's depth-based section scan cannot segment this; glosa's level stack
    must produce the same units as `nested_doc`.
    """
    doc = DoclingDocument(name="annual-report")
    _pages(doc, 2)
    doc.add_title(text="Annual Report 2025", prov=prov(1, 740))
    doc.add_heading(text="Revenue", level=1, prov=prov(1, 700))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Revenue reached 12.4M EUR, up 8% year on year.",
        prov=prov(1, 680),
    )
    doc.add_heading(text="Risks", level=1, prov=prov(2, 700))
    doc.add_heading(text="Legal", level=2, prov=prov(2, 660))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="A supplier dispute is pending before the commercial court.",
        prov=prov(2, 640),
    )
    return doc


@pytest.fixture
def headless_doc() -> DoclingDocument:
    """No headings at all — upstream returns the whole document as the answer."""
    doc = DoclingDocument(name="scan")
    _pages(doc, 2)
    doc.add_text(label=DocItemLabel.TEXT, text="First page body text.", prov=prov(1, 700))
    doc.add_text(label=DocItemLabel.TEXT, text="Second page body text.", prov=prov(2, 700))
    return doc


@pytest.fixture
def preamble_doc() -> DoclingDocument:
    """Content before the first heading must still be reachable."""
    doc = DoclingDocument(name="memo")
    _pages(doc, 1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Internal memo, circulated to the board on 4 March.",
        prov=prov(1, 740),
    )
    heading = doc.add_heading(text="Decision", level=1, prov=prov(1, 700))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="The board approved the budget.",
        parent=heading,
        prov=prov(1, 680),
    )
    return doc


class FakeChatModel:
    """Scripted `ChatModel`.

    `script` is consumed in order: a `BaseModel` answers the next `structured`
    call, a `str` answers the next `complete` call, an `Exception` is raised.
    """

    def __init__(self, script: list[Any], *, model_id: str = "fake-model") -> None:
        self.script = list(script)
        self._model_id = model_id
        self.structured_calls: list[list[Message]] = []
        self.complete_calls: list[list[Message]] = []
        self.closed = False

    @property
    def model_id(self) -> str:
        return self._model_id

    def for_model(self, model_id: str) -> FakeChatModel:
        twin = FakeChatModel(self.script, model_id=model_id)
        twin.script = self.script  # share the queue so assertions still see it
        return twin

    def _next(self) -> Any:
        if not self.script:
            raise AssertionError("FakeChatModel script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        self.complete_calls.append(messages)
        item = self._next()
        assert isinstance(item, str), f"expected a str for complete(), got {type(item).__name__}"
        return item

    async def structured[T: BaseModel](
        self, messages: list[Message], *, schema: type[T], max_tokens: int | None = None
    ) -> T:
        self.structured_calls.append(messages)
        item = self._next()
        assert isinstance(item, schema), (
            f"expected {schema.__name__} for structured(), got {type(item).__name__}"
        )
        return item

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_model_factory() -> type[FakeChatModel]:
    return FakeChatModel


def assert_is_chat_model(model: object) -> None:
    assert isinstance(model, ChatModel)
