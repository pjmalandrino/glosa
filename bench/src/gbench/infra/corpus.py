"""Reading the corpus off disk.

The one place that knows a document is a paper on arXiv, a `DoclingDocument` on
disk, and a YAML file of questions.

**What is committed and what is not.** Forty PDFs and their conversions are
hundreds of megabytes; a git repository is the wrong place for them. So:

* `manifest.yaml` — **committed.** arXiv id, version, SHA-256, licence,
  category. The corpus *is* this file.
* `items/*.yaml` — **committed.** The questions, a few kilobytes.
* `docs/<slug>/refs.json` — **committed always.** One line per projected
  element: ref, label, character count, and the SHA-256 of its folded text. No
  text. Enough for the quota, the abstract check and every structural claim;
  not enough to redistribute the paper.
* `docs/<slug>/sections.json` — **committed only under `licence_policy:
  redistributable`.** The projected text itself, which is what makes every
  quote checkable offline — and what restricts the corpus to CC-BY papers.
* `docs/<slug>/source.pdf` — not committed. `gbench fetch`, verified against
  the manifest's SHA-256.
* `docs/<slug>/docling.json`, `doc.md` — not committed. `gbench convert`.

The consequence worth stating: `lint` and `score` run in CI with nothing
downloaded, and only `run` needs the full conversion. A third party can check
every quote, every distractor and every published number without a GPU and
without re-downloading forty papers — and can still reproduce the conversions
exactly, because the manifest pins the bytes.

`sections.json` and `refs.json` are generated *from* `docling.json` and can
therefore drift from it. `tests/test_harness.py` fails if they do, wherever
both are present.

**Which policy.** `fetch-only` is the default because it is what makes the
corpus buildable at all: restricting forty papers to CC-BY rules out most of
arXiv, and hand-picking around that restriction is hours of licence-checking
before a single question is written. `redistributable` is strictly better when
the papers allow it — every quote checkable by anyone, forever, from a clone —
so a corpus that happens to be all CC-BY should say so and get the stronger
guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from glosa.infra.docling.projection import DoclingProjector

from gbench.domain.item import Item, ItemKind
from gbench.domain.lint import LicencePolicy
from gbench.ports.engine import BenchDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

ABSTRACT_TITLES = ("abstract", "résumé", "resume", "summary")
"""Headings that open a paper's own summary, folded."""


class CorpusIncomplete(RuntimeError):
    """A document's conversion is missing. Says what to run."""


@dataclass(frozen=True, slots=True)
class Paper:
    """One manifest entry — the corpus's unit of provenance."""

    slug: str
    title: str = ""
    arxiv_id: str = ""
    version: str = "v1"
    sha256: str = ""
    license: str = ""
    primary_category: str = ""
    generated: bool = False
    """True for the fixture document, which has no upstream to fetch."""

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}{self.version}"


@dataclass(frozen=True, slots=True)
class Section:
    ref: str
    text: str
    label: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(text_key(self.text).encode()).hexdigest()[:16]


def text_key(text: str) -> str:
    """What a ref's hash is taken over: whitespace folded, nothing else.

    Not the raw string — a conversion that changes one line break must not
    read as a changed paper. Not case-folded either: a hash is for detecting
    drift, and `lint --verify-quotes` does the semantic folding."""
    return " ".join(text.split())


class FileCorpus:
    """`Corpus` over a directory. Pure reads; nothing here calls a model."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._suite: dict[str, Any] = _load_yaml(root / "suite.yaml")
        self._papers = {
            entry["slug"]: Paper(
                slug=str(entry["slug"]),
                title=str(entry.get("title", "")),
                arxiv_id=str(entry.get("arxiv_id", "")),
                version=str(entry.get("version", "v1")),
                sha256=str(entry.get("sha256", "")),
                license=str(entry.get("license", "")),
                primary_category=str(entry.get("primary_category", "")),
                generated=bool(entry.get("generated", False)),
            )
            for entry in _load_yaml(root / "manifest.yaml").get("documents", [])
        }
        self._sections: dict[str, tuple[Section, ...]] = {}
        self._refs: dict[str, dict[str, dict[str, Any]]] = {}
        raw_policy = _load_yaml(root / "manifest.yaml").get("licence_policy", "fetch-only")
        self.policy = LicencePolicy(str(raw_policy))

    # --- documents ------------------------------------------------------------

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(self._papers)

    @property
    def suite(self) -> dict[str, Any]:
        return self._suite

    @property
    def papers(self) -> dict[str, Paper]:
        return dict(self._papers)

    def licenses(self) -> dict[str, str]:
        return {slug: paper.license for slug, paper in self._papers.items()}

    def dir_of(self, slug: str) -> Path:
        return self._root / "docs" / slug

    def document(self, slug: str) -> BenchDocument:
        """The full conversion — what the engines read. Needs `gbench convert`."""
        path = self.dir_of(slug) / "docling.json"
        if not path.exists():
            raise CorpusIncomplete(
                f"{slug}: no docling.json. Run `gbench fetch` then `gbench convert`."
            )
        raw = path.read_text(encoding="utf-8")
        markdown = self.dir_of(slug) / "doc.md"
        pdf = self.dir_of(slug) / "source.pdf"
        return BenchDocument(
            slug=slug,
            docling_json=raw,
            markdown=markdown.read_text(encoding="utf-8") if markdown.exists() else "",
            pdf_path=str(pdf) if pdf.exists() else None,
            doc_hash=hashlib.sha256(raw.encode()).hexdigest()[:16],
        )

    def converted(self, slug: str) -> bool:
        return (self.dir_of(slug) / "docling.json").exists()

    # --- text -----------------------------------------------------------------

    def sections(self, slug: str) -> tuple[Section, ...]:
        """The paper's projected elements — from `sections.json` when it is
        committed, from the conversion otherwise."""
        if slug in self._sections:
            return self._sections[slug]

        path = self.dir_of(slug) / "sections.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            found = tuple(
                Section(str(row["ref"]), str(row.get("text", "")), str(row.get("label", "")))
                for row in payload.get("elements", [])
            )
        elif self.converted(slug):
            found = self.project(slug)
        else:
            found = ()
        self._sections[slug] = found
        return found

    def refs(self, slug: str) -> dict[str, dict[str, Any]]:
        """ref → {label, chars, sha256}. Committed under every policy."""
        if slug not in self._refs:
            path = self.dir_of(slug) / "refs.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._refs[slug] = {str(row["ref"]): row for row in payload.get("elements", [])}
            else:
                self._refs[slug] = {
                    section.ref: {
                        "label": section.label,
                        "chars": len(section.text),
                        "sha256": section.digest,
                    }
                    for section in self.sections(slug)
                }
        return self._refs[slug]

    def has_text(self, slug: str) -> bool:
        """Whether quotes can be checked without fetching the paper."""
        return any(section.text for section in self.sections(slug))

    def project(self, slug: str) -> tuple[Section, ...]:
        """Straight from `docling.json`, through glosa's own projector.

        Refs mean what the engines will be scored against, or `hit@k` would be
        measuring the corpus rather than the engines.
        """
        projection = DoclingProjector().project(
            (self.dir_of(slug) / "docling.json").read_text(encoding="utf-8")
        )
        return tuple(
            Section(element.self_ref, element.text, element.graph_label)
            for element in projection.elements
        )

    def text_of(self, slug: str, ref: str) -> str:
        for section in self.sections(slug):
            if section.ref == ref:
                return section.text
        return ""

    def doc_text(self, slug: str) -> str:
        return "\n".join(section.text for section in self.sections(slug))

    def ref_chars(self, slug: str) -> dict[str, int]:
        """Character counts — from `refs.json` when the text is not committed."""
        return {ref: int(row.get("chars", 0)) for ref, row in self.refs(slug).items()}

    def abstract_refs(self, slug: str) -> tuple[str, ...]:
        """Title and abstract — everything before the paper's first real section.

        Two ways in, because conversions disagree: an explicit "Abstract"
        heading when there is one, and otherwise everything up to the first
        heading, which is where the front matter lives.
        """
        sections = self.sections(slug) or tuple(
            Section(ref, "", str(row.get("label", ""))) for ref, row in self.refs(slug).items()
        )
        folded = [(s, " ".join(s.text.casefold().split())) for s in sections]
        for index, (section, text) in enumerate(folded):
            if text.rstrip(" :.") in ABSTRACT_TITLES:
                out = [section.ref]
                for following, _ in folded[index + 1 :]:
                    if _is_heading(following):
                        break
                    out.append(following.ref)
                return tuple(out)

        out = []
        for index, (section, _) in enumerate(folded):
            if _is_heading(section) and index > 0:
                break
            out.append(section.ref)
        return tuple(out)

    def abstract_text(self, slug: str) -> str:
        inside = set(self.abstract_refs(slug))
        return "\n".join(s.text for s in self.sections(slug) if s.ref in inside)

    # --- items ----------------------------------------------------------------

    @cached_property
    def _items(self) -> tuple[Item, ...]:
        items: list[Item] = []
        for path in sorted((self._root / "items").glob("*.yaml")):
            for raw in _load_yaml_list(path):
                items.append(self._item_from(raw))
        return tuple(items)

    def items(self, slug: str | None = None) -> Sequence[Item]:
        if slug is None:
            return self._items
        return [item for item in self._items if item.doc == slug]

    def _item_from(self, raw: dict[str, Any]) -> Item:
        doc = str(raw["doc"])
        paper = self._papers.get(doc)
        return Item(
            id=str(raw["id"]),
            doc=doc,
            kind=ItemKind(raw.get("kind", "lookup")),
            question=str(raw["question"]),
            options={str(k): str(v) for k, v in raw["options"].items()},
            answer=str(raw["answer"]),
            # The field is the paper's, not the question's — so it is taken from
            # the manifest and cannot drift item by item.
            domain=str(raw.get("domain") or (paper.primary_category if paper else "")),
            gold_refs=tuple(raw.get("gold_refs", ()) or ()),
            gold_quote=str(raw.get("gold_quote", "") or ""),
            distractor_refs={str(k): str(v) for k, v in (raw.get("distractor_refs") or {}).items()},
            removed_from=str(raw.get("removed_from", "") or ""),
            authored_by=str(raw.get("authored_by", "") or ""),
            authored_from=str(raw.get("authored_from", "document")),
        )


HEADING_LABEL = "SectionHeader"
"""What glosa's projection calls a heading.

Both `title` and `section_header` collapse to it — Studio's `LABEL_MAP` does
that, and glosa follows Studio (`DESIGN.md` §2). Which is why `abstract_refs`
stops at the first heading *after* index 0 rather than at the first heading:
the paper's own title is one."""


def _is_heading(section: Section) -> bool:
    return section.label == HEADING_LABEL


def write_refs(path: Path, slug: str, sections: Sequence[Section]) -> None:
    """Structure without text — committed under every policy.

    What survives here is everything a reader needs to check that the corpus
    describes the paper it claims to: which refs exist, how big each is, what
    kind of node it is, and a hash that changes if the conversion does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "slug": slug,
                "elements": [
                    {
                        "ref": s.ref,
                        "label": s.label,
                        "chars": len(s.text),
                        "sha256": s.digest,
                    }
                    for s in sections
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def write_sections(path: Path, slug: str, sections: Sequence[Section]) -> None:
    """Emit the text too. Only under `redistributable`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "slug": slug,
                "elements": [{"ref": s.ref, "label": s.label, "text": s.text} for s in sections],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, list) else []
