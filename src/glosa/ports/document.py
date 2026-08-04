"""Ports for reading a document.

`DocumentProjection` is what the domain needs a document to *be*: an ordered
set of located elements, grouped into section scopes, each addressable by an id
the host's UI already knows. It says nothing about Docling, JSON, or how the
host collapses its tree — `glosa.infra.docling` supplies all of that.

`TreeReader` points the other way: a port the Docling adapter *consumes*, so
the host can hand over its own implementation of the collapse rules (Docling
Studio's `DocumentTreeReader` satisfies it as-is) and the deployment runs one
implementation instead of two that can drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from glosa.domain.values import Element, Scope


@runtime_checkable
class DocumentProjection(Protocol):
    """A parsed document, in the shape the host displays it."""

    @property
    def title(self) -> str: ...

    @property
    def elements(self) -> tuple[Element, ...]:
        """Every projected element, in reading order."""
        ...

    @property
    def node_ids(self) -> frozenset[str]:
        """Host graph ids of every element — what a trace may reference."""
        ...

    @property
    def has_sections(self) -> bool: ...

    @property
    def page_numbers(self) -> tuple[int, ...]: ...

    def readable(self, *, include_furniture: bool = False) -> tuple[Element, ...]:
        """Elements worth putting in front of a model, in reading order."""
        ...

    def scopes(self, *, include_furniture: bool = False) -> tuple[Scope, ...]:
        """Section scopes, as the host groups them."""
        ...

    def page_elements(
        self, page_no: int, *, include_furniture: bool = False
    ) -> tuple[Element, ...]: ...

    def page_ref(self, page_no: int) -> str:
        """The host's ref for a page — the domain must not invent one."""
        ...

    def page_node_id(self, page_no: int) -> str:
        """The host's graph id for a page node."""
        ...


@runtime_checkable
class DocumentProjector(Protocol):
    """Turns a stored document payload into a `DocumentProjection`."""

    def project(self, document_json: str) -> DocumentProjection:
        """Raises `DocumentParseError` if the payload is not a document."""
        ...


@runtime_checkable
class TreeReader(Protocol):
    """Walks a serialized `DoclingDocument` the way the host projects it.

    Transcribed from Docling Studio's `domain.ports.DocumentTreeReader`.
    """

    def iter_items(self, doc_data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield `(source_list_key, item)` for texts/tables/pictures/groups."""
        ...

    def is_inline_group(self, item: dict[str, Any]) -> bool:
        """True iff `item` is an InlineGroup collapsed into one projection."""
        ...

    def build_collapse_index(
        self, doc_data: dict[str, Any]
    ) -> tuple[set[str], dict[str, dict[str, Any]]]:
        """Return `(skip_refs, inline_meta)` for the projection."""
        ...
