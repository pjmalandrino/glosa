"""Docling Studio's document object — the thing glosa reads.

Structure, node ids, reading order and section scoping all come from Studio's
own projection so a trace can never point at a node the UI does not have.
"""

from glosa.studio.ports import TreeReader
from glosa.studio.projection import (
    Element,
    Scope,
    StudioProjection,
    node_id_for,
    page_node_id,
    to_topleft,
)
from glosa.studio.tree import LABEL_MAP, BundledTreeReader, element_label

__all__ = [
    "LABEL_MAP",
    "BundledTreeReader",
    "Element",
    "Scope",
    "StudioProjection",
    "TreeReader",
    "element_label",
    "node_id_for",
    "page_node_id",
    "to_topleft",
]
