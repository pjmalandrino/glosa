"""Docling adapter: serialized `DoclingDocument` → `DocumentProjection`."""

from glosa.infra.docling.projection import (
    DoclingProjection,
    DoclingProjector,
    node_id_for,
    page_node_id,
    page_ref,
    to_topleft,
)
from glosa.infra.docling.tree import LABEL_MAP, BundledTreeReader, element_label

__all__ = [
    "LABEL_MAP",
    "BundledTreeReader",
    "DoclingProjection",
    "DoclingProjector",
    "element_label",
    "node_id_for",
    "page_node_id",
    "page_ref",
    "to_topleft",
]
