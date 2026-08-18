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
PROSE_LABELS = frozenset({"text", "paragraph", "list_item", "inline"})
"""Labels whose text is running prose, quotable as-is.

Everything else — a table rendered to markup, a picture placeholder, a formula,
a code block, a caption belonging to a figure rather than to the section — is
text that reads as noise when lifted out of its context."""


def element_label(docling_label: str) -> str:
    return LABEL_MAP.get(docling_label.lower(), DEFAULT_LABEL)


def item_label(item: dict[str, Any]) -> str:
    return (item.get("label") or "").lower()


def is_inline_group(item: dict[str, Any]) -> bool:
    return item_label(item) == "inline"


def is_picture(item: dict[str, Any]) -> bool:
    return item_label(item) in {"picture", "chart"}


def is_prose(item: dict[str, Any]) -> bool:
    """True iff this item's text can be quoted as a sentence of the document."""
    return item_label(item) in PROSE_LABELS


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
    """Yield `(source_list_key, item)` for every item in texts/tables/pictures/groups.

    Anything that is not a list of dicts is skipped rather than crashed on:
    the payload is untrusted storage, and a corrupt entry must surface as a
    degraded projection or a `DocumentParseError`, never a raw `AttributeError`.
    """
    for key in ITEM_LISTS:
        items = doc_data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield key, item


def index_by_ref(doc_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    for _, item in iter_items(doc_data):
        ref = item.get("self_ref")
        if ref:
            by_ref[ref] = item
    return by_ref


def _safe_float(value: Any) -> float:
    try:
        return 0.0 if value is None else float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def iter_provs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `prov[]` into Studio's row shape, preserving order.

    Coordinates and spans are coerced defensively: a prov row with a garbage
    bbox loses its location, not the whole document.
    """
    provs = item.get("prov")
    if not isinstance(provs, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, prov in enumerate(provs):
        if not isinstance(prov, dict):
            continue
        bbox = prov.get("bbox")
        left = top = right = bottom = 0.0
        if isinstance(bbox, dict):
            left = _safe_float(bbox.get("l"))
            top = _safe_float(bbox.get("t"))
            right = _safe_float(bbox.get("r"))
            bottom = _safe_float(bbox.get("b"))
        elif isinstance(bbox, list | tuple) and len(bbox) >= 4:
            left, top, right, bottom = (_safe_float(x) for x in bbox[:4])
        coord_origin = (bbox.get("coord_origin") if isinstance(bbox, dict) else None) or "TOPLEFT"
        charspan = prov.get("charspan")
        if not isinstance(charspan, list | tuple):
            charspan = ()
        rows.append(
            {
                "order": idx,
                "page_no": prov.get("page_no"),
                "bbox_l": left,
                "bbox_t": top,
                "bbox_r": right,
                "bbox_b": bottom,
                "coord_origin": coord_origin,
                "charspan_start": _safe_int(charspan[0]) if len(charspan) >= 1 else None,
                "charspan_end": _safe_int(charspan[1]) if len(charspan) >= 2 else None,
            }
        )
    return rows


def iter_pages(doc_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield `{page_no, width, height}` for each page in the `pages` map."""
    pages = doc_data.get("pages")
    if not isinstance(pages, dict):
        return
    for page_no_str, page_obj in pages.items():
        try:
            page_no = int(page_no_str)
        except (TypeError, ValueError):
            continue
        size = page_obj.get("size") if isinstance(page_obj, dict) else None
        if not isinstance(size, dict):
            size = {}
        yield {"page_no": page_no, "width": size.get("width"), "height": size.get("height")}


def dfs_order(doc_data: dict[str, Any], skip_refs: set[str] | None = None) -> list[str]:
    """`self_ref`s in reading order — the NEXT chain the graph draws.

    Each ref is visited once: a ref listed under two parents appears at its
    first position only, and a cycle in the `children` graph terminates
    instead of recursing forever.
    """
    skip = skip_refs or set()
    by_ref = index_by_ref(doc_data)
    order: list[str] = []
    seen: set[str] = set()

    def walk(children: Any) -> None:
        if not isinstance(children, list):
            return
        for child in children:
            if not isinstance(child, dict):
                continue
            ref = child_ref(child)
            if not ref or ref in skip or ref in seen:
                continue
            seen.add(ref)
            order.append(ref)
            item = by_ref.get(ref)
            if item and not is_inline_group(item):
                walk(item.get("children"))

    body = doc_data.get("body")
    walk(body.get("children") if isinstance(body, dict) else None)
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
        children = item.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            ref_ = child_ref(child) if isinstance(child, dict) else None
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
        children = item.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            ref_ = child_ref(child) if isinstance(child, dict) else None
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
