"""Reading the corpus off disk.

The one place that knows a document is a `DoclingDocument` on disk and an item
is YAML. It resolves refs through **glosa's own projection**, which is not a
convenience: `gold_refs` must mean the same nodes the engines are scored
against, and the projection is the definition of what nodes exist
(`DESIGN.md` §2). Authoring against docling-core's raw item tree instead would
produce refs that no engine can hit.

Layout:

    corpus/
      suite.yaml                      documents, defaults, profiles
      docs/<slug>/docling.json        the committed conversion
      docs/<slug>/doc.md              its export_to_markdown(), committed
      docs/<slug>/source.pdf          kept, unused by the headline runs
      items/<slug>.yaml               the questions
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from glosa.infra.docling.projection import DoclingProjector

from gbench.domain.item import Item, ItemKind
from gbench.ports.engine import BenchDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    from glosa.ports.document import DocumentProjection


@dataclass(frozen=True, slots=True)
class _Doc:
    slug: str
    root: Path

    @property
    def docling_json(self) -> str:
        return (self.root / "docling.json").read_text(encoding="utf-8")

    @property
    def markdown(self) -> str:
        path = self.root / "doc.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @property
    def pdf(self) -> str | None:
        path = self.root / "source.pdf"
        return str(path) if path.exists() else None


class FileCorpus:
    """`Corpus` over a directory. Pure reads; nothing here calls a model."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._suite: dict[str, Any] = yaml.safe_load(
            (root / "suite.yaml").read_text(encoding="utf-8")
        )
        self._docs = {
            slug: _Doc(slug, root / "docs" / slug) for slug in self._suite.get("documents", [])
        }
        self._projections: dict[str, DocumentProjection] = {}

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(self._docs)

    @property
    def suite(self) -> dict[str, Any]:
        return self._suite

    def document(self, slug: str) -> BenchDocument:
        doc = self._docs[slug]
        raw = doc.docling_json
        return BenchDocument(
            slug=slug,
            docling_json=raw,
            markdown=doc.markdown,
            pdf_path=doc.pdf,
            doc_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
        )

    @cached_property
    def _items(self) -> tuple[Item, ...]:
        items: list[Item] = []
        for path in sorted((self._root / "items").glob("*.yaml")):
            for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
                items.append(_item_from(raw))
        return tuple(items)

    def items(self, slug: str | None = None) -> Sequence[Item]:
        if slug is None:
            return self._items
        return [item for item in self._items if item.doc == slug]

    # --- text resolution, through the projection ------------------------------

    def _projection(self, slug: str) -> DocumentProjection:
        if slug not in self._projections:
            self._projections[slug] = DoclingProjector().project(self._docs[slug].docling_json)
        return self._projections[slug]

    def text_of(self, slug: str, ref: str) -> str:
        for element in self._projection(slug).elements:
            if element.self_ref == ref:
                return element.text
        return ""

    def doc_text(self, slug: str) -> str:
        return "\n".join(element.text for element in self._projection(slug).elements)

    def ref_chars(self, slug: str) -> dict[str, int]:
        return {element.self_ref: len(element.text) for element in self._projection(slug).elements}


def _item_from(raw: dict[str, Any]) -> Item:
    return Item(
        id=str(raw["id"]),
        doc=str(raw["doc"]),
        kind=ItemKind(raw.get("kind", "lookup")),
        question=str(raw["question"]),
        options={str(k): str(v) for k, v in raw["options"].items()},
        answer=str(raw["answer"]),
        gold_refs=tuple(raw.get("gold_refs", ()) or ()),
        gold_quote=str(raw.get("gold_quote", "") or ""),
        distractor_refs={str(k): str(v) for k, v in (raw.get("distractor_refs") or {}).items()},
        removed_from=str(raw.get("removed_from", "") or ""),
        authored_by=str(raw.get("authored_by", "") or ""),
        authored_from=str(raw.get("authored_from", "document")),
    )
