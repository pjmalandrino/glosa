"""Regenerates this document's committed artefacts.

    uv run --project .. python _build.py

A fixture, not a corpus document: it exists so the harness can be exercised end
to end — lint, run against a scripted model, score — with no PDF, no GPU and no
network, in CI.

It carries one of every structure the five item kinds need: prose that states a
fact, a definition used somewhere else, a table, and something out on the edge
of the reading order (a footnote). A paper has all four too; this has them in
twenty elements instead of a thousand.

Real corpus documents are converted PDFs, pinned by `manifest.yaml` and rebuilt
with `gbench fetch` + `gbench convert`. Only this one is generated and
committed, and it says so.
"""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ProvenanceItem

HERE = Path(__file__).parent

ARTICLES: list[tuple[str, str]] = [
    (
        "Article 1 — Objet",
        "Le présent cahier des clauses administratives particulières fixe les "
        "conditions d'exécution du marché de travaux de réhabilitation.",
    ),
    (
        "Article 2 — Définitions",
        "Le montant du marché s'entend hors taxes, à l'exclusion de toute révision de prix.",
    ),
    (
        "Article 4 — Délais",
        "Le délai global d'exécution est fixé à 180 jours calendaires à compter "
        "de la notification de l'ordre de service.",
    ),
    (
        "Article 7 — Pénalités de retard",
        "Les pénalités de retard sont fixées à 1/1000 du montant du marché par "
        "jour calendaire de retard, sans mise en demeure préalable.",
    ),
    (
        "Article 9 — Retenue de garantie",
        "La retenue de garantie est fixée à 5 % du montant du marché et peut être "
        "remplacée par une garantie à première demande.",
    ),
    (
        "Article 12 — Assurances",
        "Le titulaire justifie d'une assurance de responsabilité civile d'un "
        "montant minimal de 2 000 000 EUR par sinistre.",
    ),
    (
        "Article 13 — Avance",
        "Une avance de 20 % du montant du marché est versée au titulaire dans "
        "les trente jours suivant la notification.",
    ),
    (
        "Article 15 — Résiliation",
        "Le pouvoir adjudicateur peut résilier le marché pour motif d'intérêt "
        "général, après information écrite du titulaire.",
    ),
]


def prov(page_no: int, top: float) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=72.0, t=top, r=540.0, b=top - 14.0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 10),
    )


def build() -> DoclingDocument:
    doc = DoclingDocument(name="toy-ccap")
    for page_no in (1, 2):
        doc.add_page(page_no=page_no, size=Size(width=612.0, height=792.0))

    doc.add_title(text="Marché public de travaux — CCAP", prov=prov(1, 740))
    top = 700.0
    for offset, (heading, body) in enumerate(ARTICLES):
        page = 1 if offset < 3 else 2
        doc.add_heading(text=heading, level=1, prov=prov(page, top))
        doc.add_text(label=DocItemLabel.TEXT, text=body, prov=prov(page, top - 20))
        top = top - 60 if page == 1 else 700.0 - 60 * (offset - 3)

    add_table(doc)
    # Out on the edge of the reading order — no heading owns it, and a reader
    # that walks headings from the top never arrives.
    doc.add_text(label=DocItemLabel.FOOTNOTE, text=FOOTNOTE, prov=prov(2, 120))
    return doc


GUARANTEE_SCALE = [
    ["Tranche de montant", "Retenue de garantie"],
    ["Jusqu'à 500 000 EUR", "5 % du montant du marché"],
    ["De 500 001 à 2 000 000 EUR", "4 % du montant du marché"],
    ["Au-delà de 2 000 000 EUR", "3 % du montant du marché"],
]

FOOTNOTE = (
    "Les jours calendaires comprennent les samedis, dimanches et jours fériés, "
    "sans interruption pendant la période de congés."
)


def add_table(doc: DoclingDocument) -> None:
    """A barème — where the `table` items live.

    A cell, not a paragraph: an engine that flattens the table into a line of
    text can still find the section and still lose the point."""
    cells = [
        TableCell(
            text=text,
            row_span=1,
            col_span=1,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
        )
        for row, columns in enumerate(GUARANTEE_SCALE)
        for col, text in enumerate(columns)
    ]
    doc.add_table(
        data=TableData(num_rows=len(GUARANTEE_SCALE), num_cols=2, table_cells=cells),
        prov=prov(2, 520),
    )


if __name__ == "__main__":
    document = build()
    payload = document.export_to_dict()
    (HERE / "docling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "doc.md").write_text(document.export_to_markdown(), encoding="utf-8")
    print(f"wrote {HERE / 'docling.json'} and {HERE / 'doc.md'}")
