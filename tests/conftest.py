"""Fixtures.

Documents are built with `docling-core` and handed to glosa as **serialized
JSON**, which is exactly what Docling Studio stores in
`AnalysisJob.document_json`. docling-core is a test-only dependency: glosa
itself never imports it.
"""

from __future__ import annotations

from typing import Any

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ProvenanceItem
from pydantic import BaseModel

from glosa.domain.index import DocIndex
from glosa.domain.values import Element, Scope
from glosa.infra.docling.projection import DoclingProjector
from glosa.ports.chat import ChatModel, Message

PAGE_HEIGHT = 792.0
PAGE_WIDTH = 612.0


def prov(page_no: int, top: float = 700.0, length: int = 10) -> ProvenanceItem:
    """A BOTTOMLEFT provenance — the origin Docling emits for PDFs."""
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=72.0, t=top, r=540.0, b=top - 14.0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, length),
    )


def pages(doc: DoclingDocument, count: int) -> None:
    for page_no in range(1, count + 1):
        doc.add_page(page_no=page_no, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))


def index_of(document_json: str, *, tree_reader: Any = None, **kwargs: Any) -> DocIndex:
    """Compose a projector and an index — what a host's wire-up does."""
    projection = DoclingProjector(tree_reader=tree_reader).project(document_json)
    return DocIndex(projection, **kwargs)


# --- documents ---------------------------------------------------------------


def build_nested() -> DoclingDocument:
    """Headings carry their content as children — the well-formed case."""
    doc = DoclingDocument(name="annual-report")
    pages(doc, 2)
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


def build_flat() -> DoclingDocument:
    """Same content, every item a sibling of the body — what most PDFs produce."""
    doc = DoclingDocument(name="annual-report")
    pages(doc, 2)
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


def build_nested_trailing() -> DoclingDocument:
    """A heading's own paragraph *after* a nested sub-heading's subtree.

    Reading order puts the trailing text after the `h2`'s content, but its
    explicit parent is the `h1` — and the frontend's explicit-PARENT_OF
    exemption keeps it with the `h1`. The discriminating shape for the
    scoping rule: a NEXT-chain-only reading files it under the `h2`.
    """
    doc = DoclingDocument(name="handbook")
    pages(doc, 1)
    risks = doc.add_heading(text="Risks", level=1, prov=prov(1, 740))
    doc.add_text(label=DocItemLabel.TEXT, text="Risk overview.", parent=risks, prov=prov(1, 720))
    legal = doc.add_heading(text="Legal", level=2, parent=risks, prov=prov(1, 700))
    doc.add_text(
        label=DocItemLabel.TEXT, text="A dispute is pending.", parent=legal, prov=prov(1, 680)
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Overall, risks remain manageable.",
        parent=risks,
        prov=prov(1, 660),
    )
    return doc


def build_headless() -> DoclingDocument:
    """No headings at all — upstream returns the whole document as the answer."""
    doc = DoclingDocument(name="scan")
    pages(doc, 2)
    doc.add_text(label=DocItemLabel.TEXT, text="First page body text.", prov=prov(1, 700))
    doc.add_text(label=DocItemLabel.TEXT, text="Second page body text.", prov=prov(2, 700))
    return doc


def build_preamble() -> DoclingDocument:
    """Content before the first heading must still be reachable."""
    doc = DoclingDocument(name="memo")
    pages(doc, 1)
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


def build_inline() -> DoclingDocument:
    """An InlineGroup: one `groups[]` entry plus N `texts[]` style runs.

    Studio projects this as a *single* Paragraph node; the style runs are in
    `skip_refs` and do not exist in the graph.
    """
    doc = DoclingDocument(name="styled")
    pages(doc, 1)
    doc.add_heading(text="Clause 4", level=1, prov=prov(1, 740))
    group = doc.add_inline_group()
    doc.add_text(label=DocItemLabel.TEXT, text="The fee is", parent=group, prov=prov(1, 700))
    doc.add_text(label=DocItemLabel.TEXT, text="EUR 4,500", parent=group, prov=prov(1, 700))
    doc.add_text(label=DocItemLabel.TEXT, text="per month.", parent=group, prov=prov(1, 700))
    return doc


def build_picture() -> DoclingDocument:
    """A picture whose children are labels lifted out of the diagram.

    Studio keeps the picture node and drops its descendants.
    """
    doc = DoclingDocument(name="diagram")
    pages(doc, 1)
    doc.add_heading(text="Architecture", level=1, prov=prov(1, 740))
    caption = doc.add_text(
        label=DocItemLabel.CAPTION, text="Figure 1 — deployment topology", prov=prov(1, 600)
    )
    picture = doc.add_picture(prov=prov(1, 680), caption=caption)
    doc.add_text(
        label=DocItemLabel.TEXT, text="NOISE-AXIS-LABEL", parent=picture, prov=prov(1, 670)
    )
    return doc


def build_table() -> DoclingDocument:
    doc = DoclingDocument(name="figures")
    pages(doc, 1)
    heading = doc.add_heading(text="Financials", level=1, prov=prov(1, 740))
    cells = [
        TableCell(
            text="Year",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            column_header=True,
        ),
        TableCell(
            text="Revenue",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            column_header=True,
        ),
        TableCell(
            text="2025",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
        TableCell(
            text="12.4M",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    ]
    doc.add_table(
        data=TableData(num_rows=2, num_cols=2, table_cells=cells),
        parent=heading,
        prov=prov(1, 700),
    )
    return doc


def build_furniture() -> DoclingDocument:
    """A running head: a real graph node, but noise in an excerpt."""
    from docling_core.types.doc.common.content_layer import ContentLayer

    doc = DoclingDocument(name="report")
    pages(doc, 1)
    doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="CONFIDENTIAL — Acme Corp",
        content_layer=ContentLayer.FURNITURE,
        prov=prov(1, 780),
    )
    heading = doc.add_heading(text="Summary", level=1, prov=prov(1, 740))
    doc.add_text(
        label=DocItemLabel.TEXT, text="Everything is fine.", parent=heading, prov=prov(1, 700)
    )
    return doc


@pytest.fixture
def flat_json() -> str:
    return build_flat().model_dump_json()


@pytest.fixture
def nested_json() -> str:
    return build_nested().model_dump_json()


@pytest.fixture
def headless_json() -> str:
    return build_headless().model_dump_json()


@pytest.fixture
def preamble_json() -> str:
    return build_preamble().model_dump_json()


@pytest.fixture
def inline_json() -> str:
    return build_inline().model_dump_json()


@pytest.fixture
def picture_json() -> str:
    return build_picture().model_dump_json()


@pytest.fixture
def table_json() -> str:
    return build_table().model_dump_json()


@pytest.fixture
def furniture_json() -> str:
    return build_furniture().model_dump_json()


class FakeProjection:
    """A `DocumentProjection` built by hand — no Docling, no JSON.

    Its existence is the point: the domain works against the port, so anything
    that satisfies the protocol can drive it.
    """

    def __init__(self, elements: tuple[Element, ...], *, title: str = "fake") -> None:
        self._elements = elements
        self._title = title

    @property
    def title(self) -> str:
        return self._title

    @property
    def elements(self) -> tuple[Element, ...]:
        return self._elements

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(e.node_id for e in self._elements)

    @property
    def has_sections(self) -> bool:
        return any(e.is_section for e in self._elements)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(sorted({p for e in self._elements for p in e.pages}))

    def readable(self, *, include_furniture: bool = False) -> tuple[Element, ...]:
        if include_furniture:
            return self._elements
        return tuple(e for e in self._elements if not e.is_furniture)

    def scopes(self, *, include_furniture: bool = False) -> tuple[Scope, ...]:
        scopes: list[Scope] = []
        anchor: Element | None = None
        members: list[Element] = []
        for element in self.readable(include_furniture=include_furniture):
            if element.is_section:
                if anchor is not None:
                    scopes.append(Scope(anchor, tuple(members)))
                elif members:
                    scopes.append(Scope(members[0], tuple(members[1:])))
                anchor, members = element, []
                continue
            members.append(element)
        if anchor is not None:
            scopes.append(Scope(anchor, tuple(members)))
        elif members:
            scopes.append(Scope(members[0], tuple(members[1:])))
        return tuple(scopes)

    def page_elements(
        self, page_no: int, *, include_furniture: bool = False
    ) -> tuple[Element, ...]:
        return tuple(
            e for e in self.readable(include_furniture=include_furniture) if page_no in e.pages
        )

    def page_ref(self, page_no: int) -> str:
        return f"page/{page_no}"

    def page_node_id(self, page_no: int) -> str:
        return f"p::{page_no}"


# --- scripted model ----------------------------------------------------------


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


def assert_is_chat_model(model: object) -> None:
    assert isinstance(model, ChatModel)
