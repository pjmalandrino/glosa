"""Section scoping must agree with the frontend, node for node.

`frontend/src/features/analysis/sectionParenting.ts` decides which nodes the UI
shows inside a section. It walks the NEXT chain: every `SectionHeader` becomes
the current section, and every following node — unless it has an explicit
`PARENT_OF` — is attributed to it.

If glosa scoped sections any other way (say, by nesting `h2` inside `h1`), a
step reporting "I read #/texts/7" would highlight a different set of nodes than
the one actually read. The oracle below is that TypeScript function transcribed
to Python; the test asserts glosa's units match it exactly.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from glosa.infra.docling.projection import node_id_for
from glosa.infra.docling.tree import (
    build_collapse_index,
    dfs_order,
    element_label,
    item_label,
    iter_items,
    parent_ref,
)
from tests.conftest import build_flat, build_nested, build_picture, build_preamble, index_of

SECTION_LABEL = "SectionHeader"


# --- oracle: Studio's graph payload + the frontend's parenting rule ----------


def _graph(doc_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribed from `infra/docling_graph.py::build_graph_payload`."""
    skip_refs, _ = build_collapse_index(doc_data)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    kept: set[str] = set()

    for _, item in iter_items(doc_data):
        ref = item.get("self_ref")
        if not ref or ref in skip_refs:
            continue
        kept.add(ref)
        nodes.append(
            {"id": node_id_for(ref), "group": "element", "label": element_label(item_label(item))}
        )
        parent = parent_ref(item)
        if parent and parent != "#/body":
            edges.append(
                {"source": node_id_for(parent), "target": node_id_for(ref), "type": "PARENT_OF"}
            )

    for a, b in pairwise(dfs_order(doc_data, skip_refs)):
        if a in kept and b in kept:
            edges.append({"source": node_id_for(a), "target": node_id_for(b), "type": "NEXT"})
    return nodes, edges


def _compute_section_parents(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, str]:
    """Transcribed from `sectionParenting.ts::computeSectionParents`."""
    element_ids = {n["id"] for n in nodes if n["group"] == "element"}
    kind_by_id = {n["id"]: n.get("label") for n in nodes if n["group"] == "element"}
    if not element_ids:
        return {}

    explicit_parent_of: set[str] = set()
    next_map: dict[str, str] = {}
    has_incoming_next: set[str] = set()
    for edge in edges:
        if edge["source"] not in element_ids or edge["target"] not in element_ids:
            continue
        if edge["type"] == "PARENT_OF":
            explicit_parent_of.add(edge["target"])
        elif edge["type"] == "NEXT":
            next_map.setdefault(edge["source"], edge["target"])
            has_incoming_next.add(edge["target"])

    heads = sorted(i for i in element_ids if i not in has_incoming_next)
    parents: dict[str, str] = {}
    visited: set[str] = set()

    for head in heads:
        current_section: str | None = None
        node: str | None = head
        while node and node not in visited:
            visited.add(node)
            if kind_by_id.get(node) == SECTION_LABEL:
                current_section = node
            elif current_section and node not in explicit_parent_of:
                parents[node] = current_section
            node = next_map.get(node)
    return parents


def _resolve_sections(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, str | None]:
    """Section each node ends up *inside* in the UI, following compound nesting.

    A `list_item` keeps its explicit `PARENT_OF` to its `list`, which is itself
    attributed to a section — so visually it sits inside that section too.
    """
    synthetic = _compute_section_parents(nodes, edges)
    explicit = {e["target"]: e["source"] for e in edges if e["type"] == "PARENT_OF"}
    kind_by_id = {n["id"]: n.get("label") for n in nodes if n["group"] == "element"}

    def resolve(node_id: str, depth: int = 0) -> str | None:
        if depth > 64:  # pragma: no cover - cycle guard
            return None
        if kind_by_id.get(node_id) == SECTION_LABEL:
            return node_id
        if node_id in synthetic:
            return synthetic[node_id]
        parent = explicit.get(node_id)
        return resolve(parent, depth + 1) if parent else None

    return {node_id: resolve(node_id) for node_id in kind_by_id}


# --- the contract ------------------------------------------------------------


def _assert_scoping_matches(document_json: str) -> None:
    import json

    doc_data = json.loads(document_json)
    nodes, edges = _graph(doc_data)
    expected = _resolve_sections(nodes, edges)

    index = index_of(document_json, include_furniture=True)
    actual: dict[str, str | None] = {}
    for unit in index.units:
        for ref in unit.element_refs:
            actual[node_id_for(ref)] = unit.node_id

    for node_id, section_id in expected.items():
        if section_id is None:
            # Content before the first heading: the UI leaves it unparented,
            # glosa gives it a unit of its own so it stays reachable. What must
            # hold is that it is never claimed by a section.
            assert actual.get(node_id) not in {
                n["id"] for n in nodes if n.get("label") == SECTION_LABEL
            }
            continue
        assert actual.get(node_id) == section_id, (
            f"{node_id} belongs to {section_id} in the UI but to {actual.get(node_id)} in glosa"
        )


def test_scoping_matches_the_frontend_on_a_flat_document() -> None:
    _assert_scoping_matches(build_flat().model_dump_json())


def test_scoping_matches_the_frontend_on_a_nested_document() -> None:
    _assert_scoping_matches(build_nested().model_dump_json())


def test_scoping_matches_the_frontend_with_a_preamble() -> None:
    _assert_scoping_matches(build_preamble().model_dump_json())


def test_scoping_matches_the_frontend_with_a_figure() -> None:
    _assert_scoping_matches(build_picture().model_dump_json())


def test_an_h2_after_an_h1_opens_a_new_scope_it_does_not_nest() -> None:
    """The rule that most obviously differs from a level-stack reading."""
    document_json = build_flat().model_dump_json()
    index = index_of(document_json)

    risks = next(u for u in index.units if u.title == "Risks")
    legal = next(u for u in index.units if u.title == "Legal")

    assert legal.ref not in risks.element_refs
    assert set(risks.element_refs).isdisjoint(legal.element_refs)
