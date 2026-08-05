"""Where documents and items come from.

Split from `Engine` because the linter and the scorer need the corpus without
needing an engine — `gbench lint` and `gbench score` run with no model, no GPU
and no competitor installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gbench.domain.item import Item
    from gbench.ports.engine import BenchDocument


@runtime_checkable
class Corpus(Protocol):
    @property
    def slugs(self) -> tuple[str, ...]: ...

    def document(self, slug: str) -> BenchDocument: ...

    def items(self, slug: str | None = None) -> Sequence[Item]:
        """All items, or one document's."""
        ...

    def text_of(self, slug: str, ref: str) -> str:
        """The text of one projected node — what the linter checks quotes against.

        Empty when the ref does not exist, so a stale `gold_ref` is a finding
        rather than a crash.
        """
        ...

    def doc_text(self, slug: str) -> str: ...

    def ref_chars(self, slug: str) -> dict[str, int]:
        """Ref → character count, for `read_precision`."""
        ...
