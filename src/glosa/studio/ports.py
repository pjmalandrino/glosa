"""Ports glosa consumes from its host.

`TreeReader` is a transcription of Docling Studio's `domain.ports
.DocumentTreeReader`. Studio's `DoclingTreeReader` satisfies it as-is, so the
deployment can run with exactly one implementation of the InlineGroup /
picture collapse rules instead of two that can drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator


@runtime_checkable
class TreeReader(Protocol):
    """Walks a serialized `DoclingDocument` the way the host projects it."""

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
