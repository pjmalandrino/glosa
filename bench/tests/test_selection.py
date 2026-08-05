"""Choosing the papers — the part of a benchmark most likely to be quietly wrong.

Driven through the `opener` seam with a canned OAI-PMH response, so this runs
in CI with no network. What it pins is the one rule that decides whether the
corpus can be published at all: a paper under arXiv's default licence lets
arXiv redistribute it, not us, and a number computed over a paper nobody else
can read is a number nobody can check.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pytest

from gbench.infra.arxiv import Candidate, licence_of, parse_list, propose

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
ARXIV_DEFAULT = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def record(arxiv_id: str, title: str, license_url: str, categories: str) -> str:
    licence = f"<license>{license_url}</license>" if license_url else ""
    return f"""
      <record><metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
        <id>{arxiv_id}</id>
        <title>{title}</title>
        <categories>{categories}</categories>
        {licence}
        <version version="v2"/>
      </arXiv></metadata></record>"""


def feed(*records: str) -> bytes:
    body = "".join(records)
    return f"""<?xml version="1.0"?>
      <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
        <ListRecords>{body}<resumptionToken></resumptionToken></ListRecords>
      </OAI-PMH>""".encode()


@dataclass
class FakeOpener:
    payload: bytes

    def __call__(self, request: object) -> io.BytesIO:
        return io.BytesIO(self.payload)


def test_only_redistributable_papers_are_proposed():
    opener = FakeOpener(
        feed(
            record("2603.00001", "A useful paper", ARXIV_DEFAULT, "cs.CL cs.AI"),
            record("2603.00002", "An open paper", CC_BY, "cs.CL"),
            record("2603.00003", "No licence at all", "", "cs.LG"),
        )
    )
    chosen = propose(sets=["cs"], since="2026-03-01", until="2026-03-02", per_set=5, opener=opener)
    assert [c.arxiv_id for c in chosen] == ["2603.00002"]


def test_the_primary_category_becomes_the_domain_column():
    opener = FakeOpener(feed(record("2603.00010", "Open", CC_BY, "q-bio.NC physics.bio-ph")))
    (chosen,) = propose(
        sets=["q-bio"], since="2026-03-01", until="2026-03-02", per_set=1, opener=opener
    )
    assert chosen.primary_category == "q-bio.NC"
    assert chosen.version == "v2"  # the latest, not v1 — the bytes we will pin


def test_per_set_caps_the_harvest():
    opener = FakeOpener(
        feed(*(record(f"2603.0010{i}", f"Open {i}", CC_BY, "cs.CL") for i in range(6)))
    )
    chosen = propose(sets=["cs"], since="2026-03-01", until="2026-03-02", per_set=5, opener=opener)
    assert len(chosen) == 5


def test_the_slug_is_stable_and_filesystem_safe():
    candidate = Candidate("2603.00002", "v1", "Attention: Is All You Need?", CC_BY, "cs.CL")
    assert candidate.slug == "2603-00002-attention-is-all-you-need"


# --- the hand-picked path ------------------------------------------------------


def test_a_pasted_list_becomes_candidates():
    # Whatever a person actually pastes: bare ids, both URL shapes, versions,
    # a category after each, and comments so the themes stay next to the papers.
    listing = """
    # 1 — NLP: dense result tables
    2603.01234  cs.CL
    https://arxiv.org/abs/2603.05678  cs.CL

    # 2 — statistical theory
    arxiv.org/pdf/2604.09999v2  math.ST
    2603.01234                      # already listed above
    """
    found = parse_list(listing)
    assert [(c.arxiv_id, c.version, c.primary_category) for c in found] == [
        ("2603.01234", "v1", "cs.CL"),
        ("2603.05678", "v1", "cs.CL"),
        ("2604.09999", "v2", "math.ST"),
    ]


def test_a_line_with_no_id_is_an_error_not_a_silent_skip():
    # Forty lines pasted by hand will contain a typo, and dropping it quietly
    # means a corpus one paper short that nobody notices.
    with pytest.raises(ValueError, match="no arXiv id"):
        parse_list("Attention Is All You Need\n")


def test_a_hand_picked_paper_needs_no_licence_up_front():
    (candidate,) = parse_list("2603.01234 cs.CL")
    assert candidate.license == ""  # `gbench fetch` fills it; fetch-only tolerates blank
    assert candidate.slug == "2603-01234"


def test_the_licence_is_read_off_the_abstract_page():
    page = b'<html><a href="http://creativecommons.org/licenses/by/4.0/">CC BY</a></html>'
    assert licence_of("2603.01234", opener=FakeOpener(page)).endswith("by/4.0/")


def test_an_unreadable_abstract_page_costs_a_field_not_the_run():
    def boom(request: object) -> None:
        raise OSError("network is not a given")

    assert licence_of("2603.01234", opener=boom) == ""
