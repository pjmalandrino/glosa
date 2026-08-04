"""Docling-shape helpers, mirroring Docling Studio's `infra/docling_tree.py`.

**This module is a mirror, not an invention.** Studio calls that module "the
single source of truth for how we read Docling's own structure"; glosa has to
read the document the *same* way or its trace points at nodes the UI does not
have. Two collapses matter (Studio issue #197):

* an **InlineGroup** — one `groups[]` entry plus N `texts[]` style runs — is
  projected as a single node carrying the concatenated text and the union of
  the runs' provenances. The style runs themselves are dropped;
* a **picture/chart**'s descendants — text labels lifted out of a diagram — are
  dropped, while the picture node stays.

So `#/texts/42` may be a ref that exists in the `DoclingDocument` and **not** in
Studio's graph. Emitting it in a trace highlights nothing. Everything glosa
surfaces goes through `build_collapse_index` first.

`tests/contract/test_studio_tree_parity.py` pins the behaviour. When Studio's
module changes, that test is what should fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ITEM_LISTS = ("texts", "tables", "pictures", "groups")

# Docling label -> Cytoscape/Neo4j label. Copied from Studio so a trace can be
# filtered by the same legend the graph view uses.
LABEL_MAP: dict[str, str] = {
    "section_header": "SectionHeader",
    "title": "SectionHeader",
    "paragraph": "Paragraph",
    "text": "Paragraph",
    "list_item": "ListItem",
    "list": "List",
    "inline": "Paragraph",
    "table": "Table",
    "picture": "Figure",
    "formula": "Formula",
    "code": "Code",
    "caption": "Caption",
    "footnote": "Footnote",
    "page_header": "PageHeader",
    "page_footer": "PageFooter",
    "key_value_area": "KeyValueArea",
    "form_area": "FormArea",
    "document_index": "DocumentIndex",
}
DEFAULT_LABEL = "TextElement"

SECTION_LABEL = "SectionHeader"
FURNITURE_LABELS = frozenset({"page_header", "page_footer"})


def element_label(docling_label: str) -> str:
    return LABEL_MAP.get(docling_label.lower(), DEFAULT_LABEL)


def item_label(item: dict[str, Any]) -> str:
    return (item.get("label") or "").lower()


def is_inline_group(item: dict[str, Any]) -> bool:
    return item_label(item) == "inline"


def is_picture(item: dict[str, Any]) -> bool:
    return item_label(item) in {"picture", "chart"}


def is_section_header(item: dict[str, Any]) -> bool:
    """True iff the graph would label this node `SectionHeader`.

    Both `title` and `section_header` map to it — which is exactly the boundary
    the frontend's `computeSectionParents` uses to scope a section.
    """
    return element_label(item_label(item)) == SECTION_LABEL


def child_ref(child: dict[str, Any]) -> str | None:
    """A child pointer is `{"$ref": ...}` or `{"cref": ...}` depending on how
    the document was serialized. Studio accepts both; so do we."""
    return child.get("$ref") or child.get("cref")


def parent_ref(item: dict[str, Any]) -> str | None:
    parent = item.get("parent")
    if isinstance(parent, dict):
        return child_ref(parent)
    return None


def iter_items(doc_data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield `(source_list_key, item)` for every item in texts/tables/pictures/groups."""
    for key in ITEM_LISTS:
        for item in doc_data.get(key, []) or []:
            yield key, item


def index_by_ref(doc_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    for _, item in iter_items(doc_data):
        ref = item.get("self_ref")
        if ref:
            by_ref[ref] = item
    return by_ref


def iter_provs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `prov[]` into Studio's row shape, preserving order."""
    rows: list[dict[str, Any]] = []
    for idx, prov in enumerate(item.get("prov") or []):
        bbox = prov.get("bbox")
        left = top = right = bottom = 0.0
        if isinstance(bbox, dict):
            left = float(bbox.get("l", 0.0) or 0.0)
            top = float(bbox.get("t", 0.0) or 0.0)
            right = float(bbox.get("r", 0.0) or 0.0)
            bottom = float(bbox.get("b", 0.0) or 0.0)
        elif isinstance(bbox, list | tuple) and len(bbox) >= 4:
            left, top, right, bottom = (float(x) for x in bbox[:4])
        coord_origin = (bbox.get("coord_origin") if isinstance(bbox, dict) else None) or "TOPLEFT"
        charspan = prov.get("charspan") or []
        rows.append(
            {
                "order": idx,
                "page_no": prov.get("page_no"),
                "bbox_l": left,
                "bbox_t": top,
                "bbox_r": right,
                "bbox_b": bottom,
                "coord_origin": coord_origin,
                "charspan_start": int(charspan[0]) if len(charspan) >= 1 else None,
                "charspan_end": int(charspan[1]) if len(charspan) >= 2 else None,
            }
        )
    return rows


def iter_pages(doc_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield `{page_no, width, height}` for each page in the `pages` map."""
    for page_no_str, page_obj in (doc_data.get("pages") or {}).items():
        try:
            page_no = int(page_no_str)
        except (TypeError, ValueError):
            continue
        size = (page_obj or {}).get("size") or {}
        yield {"page_no": page_no, "width": size.get("width"), "height": size.get("height")}


def dfs_order(doc_data: dict[str, Any], skip_refs: set[str] | None = None) -> list[str]:
    """`self_ref`s in reading order — the NEXT chain the graph draws."""
    skip = skip_refs or set()
    by_ref = index_by_ref(doc_data)
    order: list[str] = []

    def walk(children: list[dict[str, Any]] | None) -> None:
        if not children:
            return
        for child in children:
            ref = child_ref(child)
            if not ref or ref in skip:
                continue
            order.append(ref)
            item = by_ref.get(ref)
            if item and not is_inline_group(item):
                walk(item.get("children"))

    walk((doc_data.get("body") or {}).get("children"))
    return order


def build_collapse_index(
    doc_data: dict[str, Any],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return `(skip_refs, inline_meta)` — see the module docstring."""
    by_ref = index_by_ref(doc_data)
    skip_refs: set[str] = set()
    inline_meta: dict[str, dict[str, Any]] = {}

    for item in by_ref.values():
        ref = item.get("self_ref") or ""
        if not ref:
            continue
        if is_inline_group(item):
            text_parts, provs = _collect_inline_descendants(ref, by_ref, skip_refs)
            for idx, prov in enumerate(provs):
                prov["order"] = idx
            inline_meta[ref] = {"text": " ".join(text_parts), "provs": provs}
        elif is_picture(item):
            _collect_descendants(ref, by_ref, skip_refs)

    return skip_refs, inline_meta


def _collect_descendants(
    root_ref: str, by_ref: dict[str, dict[str, Any]], skip_refs: set[str]
) -> None:
    def walk(ref: str) -> None:
        item = by_ref.get(ref)
        if item is None:
            return
        for child in item.get("children") or []:
            ref_ = child_ref(child)
            if not ref_ or ref_ in skip_refs:
                continue
            skip_refs.add(ref_)
            walk(ref_)

    walk(root_ref)


def _collect_inline_descendants(
    group_ref: str, by_ref: dict[str, dict[str, Any]], skip_refs: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    text_parts: list[str] = []
    provs: list[dict[str, Any]] = []

    def walk(ref: str) -> None:
        item = by_ref.get(ref)
        if item is None:
            return
        for child in item.get("children") or []:
            ref_ = child_ref(child)
            if not ref_ or ref_ in skip_refs:
                continue
            skip_refs.add(ref_)
            node = by_ref.get(ref_)
            if node is None:
                continue
            if is_inline_group(node):
                walk(ref_)
                continue
            text = node.get("text") or ""
            if text:
                text_parts.append(text)
            provs.extend(iter_provs(node))

    walk(group_ref)
    return text_parts, provs


class BundledTreeReader:
    """Default `TreeReader`, used when the host does not inject its own.

    Studio should pass its `DoclingTreeReader` instead, so the deployment has a
    single implementation of the collapse rules.
    """

    def iter_items(self, doc_data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter_items(doc_data)

    def is_inline_group(self, item: dict[str, Any]) -> bool:
        return is_inline_group(item)

    def build_collapse_index(
        self, doc_data: dict[str, Any]
    ) -> tuple[set[str], dict[str, dict[str, Any]]]:
        return build_collapse_index(doc_data)
