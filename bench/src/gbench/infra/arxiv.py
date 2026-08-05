"""Building the corpus: choose forty papers, fetch them, pin the bytes.

Separated from everything else because it is the only part that touches the
network, and because the choice of papers is the part of a benchmark most
likely to be quietly wrong.

**Why the licence query is not optional.** A published number is only checkable
if a third party can read the paper it was computed over. arXiv's default
licence lets *arXiv* distribute the paper, not us — so the corpus is restricted
to CC-BY / CC-BY-SA / CC0, which is a real constraint on selection and rules
out plenty of good test material. The OAI-PMH interface is used rather than the
public search API because it is the one that reports the licence per record.

**Why recency is a selection criterion.** The model has very likely read a
famous paper, and a benchmark built on arXiv without guarding against that is
measuring memorisation. `propose` takes a date window; use one after the
model's training cutoff, and let the closed-book control confirm it worked.

**Why the SHA-256 is in the manifest.** arXiv versions are mutable in practice
(a v2 replaces the PDF you scored against). Pinning the bytes means a run from
six months ago and a run today either read the same paper or fail loudly.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

OAI = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
REDISTRIBUTABLE = ("creativecommons.org/licenses/by", "creativecommons.org/publicdomain")
POLITE_DELAY_S = 4.0
"""arXiv asks for one request every few seconds and enforces it with 503s."""


@dataclass(frozen=True, slots=True)
class Candidate:
    arxiv_id: str
    version: str
    title: str
    license: str
    primary_category: str

    @property
    def slug(self) -> str:
        stem = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")
        return f"{self.arxiv_id.replace('.', '-')}-{stem[:40]}".strip("-")


def propose(
    *,
    sets: Sequence[str],
    since: str,
    until: str,
    per_set: int,
    opener: object = None,
) -> list[Candidate]:
    """Harvest OAI-PMH and keep the first `per_set` redistributable papers per set.

    `sets` are arXiv OAI sets (`cs`, `math`, `physics:cond-mat`, `q-bio`, …).
    One per field gives the per-domain breakdown its columns; forty papers over
    eight fields is five each.
    """
    out: list[Candidate] = []
    for oai_set in sets:
        kept = 0
        for candidate in _harvest(oai_set, since, until, opener=opener):
            if not _redistributable(candidate.license):
                continue
            out.append(candidate)
            kept += 1
            if kept >= per_set:
                break
    return out


def _harvest(oai_set: str, since: str, until: str, *, opener: object = None) -> Iterator[Candidate]:
    token: str | None = None
    while True:
        if token:
            query = {"verb": "ListRecords", "resumptionToken": token}
        else:
            query = {
                "verb": "ListRecords",
                "metadataPrefix": "arXiv",
                "set": oai_set,
                "from": since,
                "until": until,
            }
        payload = _get(f"{OAI}?{urllib.parse.urlencode(query)}", opener=opener)
        root = ET.fromstring(payload)

        for record in root.iterfind(".//oai:record/oai:metadata/arxiv:arXiv", NS):
            found = _candidate(record)
            if found is not None:
                yield found

        element = root.find(".//oai:resumptionToken", NS)
        token = (element.text or "").strip() if element is not None else ""
        if not token:
            return
        time.sleep(POLITE_DELAY_S)


def _candidate(record: ET.Element) -> Candidate | None:
    def text(tag: str) -> str:
        found = record.find(f"arxiv:{tag}", NS)
        return (found.text or "").strip() if found is not None else ""

    arxiv_id = text("id")
    if not arxiv_id:
        return None
    versions = record.findall("arxiv:version", NS)
    version = versions[-1].get("version", "v1") if versions else "v1"
    return Candidate(
        arxiv_id=arxiv_id,
        version=version,
        title=" ".join(text("title").split()),
        license=text("license"),
        primary_category=text("categories").split(" ")[0],
    )


def _redistributable(license_url: str) -> bool:
    return any(token in license_url.lower() for token in REDISTRIBUTABLE)


def fetch(candidate_url: str, target: Path, *, expected_sha256: str = "") -> str:
    """Download a PDF and return its SHA-256, refusing a mismatch.

    A mismatch is not a warning: it means the paper under the manifest's id is
    no longer the paper the questions were authored against, and every item on
    it is now unverifiable.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _get(candidate_url, binary=True)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"{candidate_url}: sha256 {digest} does not match the manifest's "
            f"{expected_sha256} — the paper changed under its own id"
        )
    target.write_bytes(payload)
    return digest


def _get(url: str, *, binary: bool = False, opener: object = None) -> bytes:
    """`opener` is the seam the tests drive: no network in CI, ever."""
    _ = binary
    request = urllib.request.Request(url, headers={"User-Agent": "gbench (glosa benchmark)"})
    open_url = urllib.request.urlopen if not callable(opener) else opener
    with open_url(request) as response:
        payload: bytes = response.read()
    return payload
