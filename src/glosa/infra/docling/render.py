"""Element → text, on the projected document.

Rendering happens on the same node the graph shows, so an InlineGroup reads as
one paragraph (not as its style runs) and a picture reads as its caption (not
as the labels lifted out of the diagram). Tables are rendered as HTML: markdown
flattening loses row/column association, which is precisely what a numeric
question needs.
"""

from __future__ import annotations

from html import escape
from typing import Any

from glosa.infra.docling.tree import child_ref, item_label

MAX_TABLE_CELLS = 4_000


def render_item(
    item: dict[str, Any],
    *,
    by_ref: dict[str, dict[str, Any]],
    inline_text: str | None = None,
) -> str:
    """Text for one projected element."""
    if inline_text is not None:
        return inline_text.strip()

    label = item_label(item)
    if label == "table":
        return render_table(item, by_ref=by_ref)
    if label in {"picture", "chart"}:
        return render_picture(item, by_ref=by_ref)
    return str(item.get("text") or "").strip()


def _referenced_texts(refs: Any, by_ref: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    if not isinstance(refs, list):
        return out
    for entry in refs:
        if not isinstance(entry, dict):
            continue
        ref = child_ref(entry)
        if not ref:
            continue
        target = by_ref.get(ref)
        text = str((target or {}).get("text") or "").strip()
        if text:
            out.append(text)
    return out


def render_picture(item: dict[str, Any], *, by_ref: dict[str, dict[str, Any]]) -> str:
    parts = _referenced_texts(item.get("captions"), by_ref)
    for annotation in item.get("annotations") or []:
        if isinstance(annotation, dict):
            text = annotation.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    body = " ".join(parts)
    return f"[figure] {body}" if body else "[figure]"


def render_table(item: dict[str, Any], *, by_ref: dict[str, dict[str, Any]]) -> str:
    """Render a Docling table as HTML from its cell offsets.

    Cells carry `start/end_row_offset_idx` and `start/end_col_offset_idx`, so
    the grid is reconstructed exactly — including spans — without needing
    docling-core at runtime.
    """
    data = item.get("data") or {}
    cells = data.get("table_cells") or []
    caption = " ".join(_referenced_texts(item.get("captions"), by_ref))

    if not cells:
        return f"[table] {caption}".strip()

    num_rows = int(data.get("num_rows") or 0)
    num_cols = int(data.get("num_cols") or 0)
    if num_rows <= 0 or num_cols <= 0:
        for cell in cells:
            num_rows = max(num_rows, int(cell.get("end_row_offset_idx") or 0))
            num_cols = max(num_cols, int(cell.get("end_col_offset_idx") or 0))
    if num_rows * num_cols > MAX_TABLE_CELLS:
        return _render_table_rows_only(cells, caption)

    placed: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    for cell in cells:
        row = int(cell.get("start_row_offset_idx") or 0)
        col = int(cell.get("start_col_offset_idx") or 0)
        if (row, col) in placed:
            continue
        placed[(row, col)] = cell
        row_span = max(1, int(cell.get("end_row_offset_idx") or row + 1) - row)
        col_span = max(1, int(cell.get("end_col_offset_idx") or col + 1) - col)
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                if (r, c) != (row, col):
                    covered.add((r, c))

    lines: list[str] = ["<table>"]
    if caption:
        lines.append(f"<caption>{escape(caption)}</caption>")
    for row in range(num_rows):
        lines.append("<tr>")
        for col in range(num_cols):
            if (row, col) in covered:
                continue
            cell = placed.get((row, col))
            if cell is None:
                lines.append("<td></td>")
                continue
            lines.append(_render_cell(cell, row, col))
        lines.append("</tr>")
    lines.append("</table>")
    return "".join(lines)


def _render_cell(cell: dict[str, Any], row: int, col: int) -> str:
    tag = "th" if cell.get("column_header") or cell.get("row_header") else "td"
    row_span = max(1, int(cell.get("end_row_offset_idx") or row + 1) - row)
    col_span = max(1, int(cell.get("end_col_offset_idx") or col + 1) - col)
    attrs = ""
    if row_span > 1:
        attrs += f' rowspan="{row_span}"'
    if col_span > 1:
        attrs += f' colspan="{col_span}"'
    return f"<{tag}{attrs}>{escape(str(cell.get('text') or '').strip())}</{tag}>"


def _render_table_rows_only(cells: list[dict[str, Any]], caption: str) -> str:
    """Degenerate fallback for absurdly large grids: one line per cell, in order.

    Announced as such — a silently reshaped table would be worse than a plain one.
    """
    body = " | ".join(str(c.get("text") or "").strip() for c in cells[:MAX_TABLE_CELLS])
    marker = "[table too large to lay out; cells listed in order]"
    return f"{marker} {caption} {body}".strip()
