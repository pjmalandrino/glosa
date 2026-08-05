"""Regenerates this document's committed artefacts.

    uv run --project .. python _build.py

A fixture, not a corpus document: it exists so the harness can be exercised end
to end — lint, run against a scripted model, score — with no PDF, no GPU and no
network, in CI. It has the shape that matters (numbered articles, no
descriptive headings, figures scattered across sections) and none of the size.

Real corpus documents are converted PDFs and their `docling.json` is committed
verbatim; only this one is generated, and it says so.
"""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument
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
    return doc


if __name__ == "__main__":
    document = build()
    payload = document.export_to_dict()
    (HERE / "docling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "doc.md").write_text(document.export_to_markdown(), encoding="utf-8")
    print(f"wrote {HERE / 'docling.json'} and {HERE / 'doc.md'}")
