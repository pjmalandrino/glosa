"""The shipped entry point: arguments, output, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glosa.cli import main
from glosa.domain.reading import Reading
from tests.conftest import FakeChatModel


@pytest.fixture
def document(tmp_path: Path, flat_json: str) -> Path:
    path = tmp_path / "analysis.json"
    path.write_text(flat_json, encoding="utf-8")
    return path


def test_map_prints_the_outline_and_exits_zero(document: Path, capsys) -> None:
    assert main(["map", "--document", str(document)]) == 0
    out = capsys.readouterr().out
    assert "#/texts/" in out
    assert "Revenue" in out


def test_a_missing_document_file_exits_2(capsys) -> None:
    assert main(["map", "--document", "/no/such/file.json"]) == 2
    assert "error:" in capsys.readouterr().err


def test_a_payload_that_is_not_a_document_exits_4(tmp_path: Path, capsys) -> None:
    path = tmp_path / "junk.json"
    path.write_text("[]", encoding="utf-8")
    assert main(["map", "--document", str(path)]) == 4
    assert "error:" in capsys.readouterr().err


def _scripted(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> FakeChatModel:
    model = FakeChatModel(script)  # type: ignore[arg-type]
    monkeypatch.setattr("glosa.cli._model_for", lambda args: model)
    return model


def test_ask_exits_zero_when_the_document_answers(
    document: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _scripted(monkeypatch, [Reading(sufficient=True, response="12.4M EUR.")])
    assert main(["ask", "--document", str(document), "--query", "revenue?"]) == 0
    assert "12.4M EUR." in capsys.readouterr().out


def test_ask_json_carries_the_same_exit_code_as_the_outcome(
    document: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`--json` is the scripting mode; its exit code must carry the signal."""
    _scripted(monkeypatch, [Reading(sufficient=False, response="Not covered.", absent=True)])
    code = main(["ask", "--json", "--document", str(document), "--query", "cats?"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "not_in_document"
