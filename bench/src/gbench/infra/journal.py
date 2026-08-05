"""Append-only JSONL: what ran, verbatim.

`run` writes here and `score` reads here, and nothing else crosses between
them. That separation is the reason no number in this repository is one that
somebody cannot recompute: fixing a letter parser, adding a metric or changing
an abstention policy re-scores an existing journal instead of re-running a
model. It is also `DESIGN.md` §6.17 — deterministic replay — for the bench's
own runs.

One row per `(item, engine, rotation)`. Rows are never rewritten; a re-run
appends under a new `run_id`, and `score` takes the latest run per engine
unless told otherwise. Journals are committed, so a third party can re-score
this repository's claims without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gbench.domain.item import RenderedItem
    from gbench.ports.engine import EngineAnswer, PrepareCost

SCHEMA = 1


def config_hash(payload: dict[str, Any]) -> str:
    """Stable id for a run configuration — model, budget, rotations, corpus.

    Two rows with different config hashes are not comparable, and `score`
    refuses to mix them without `--allow-mixed`."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Row:
    """One scored-able observation, as stored."""

    run_id: str
    config: str
    engine: str
    signal: str
    """The engine's declared abstention mapping, as it was at run time.

    Recorded rather than looked up at scoring time: `score` must work on a
    machine where the engine cannot even be imported, and the mapping that
    applied is a property of the run, not of today's code."""
    doc: str
    item_id: str
    rotation: int
    gold: str
    text: str
    abstained: bool
    read_refs: tuple[str, ...]
    llm_calls: int | None
    prompt_chars: int | None
    wall_s: float
    contended: bool
    error: str
    grounded: bool | None
    options: dict[str, str]
    kind: str
    trace: dict[str, Any]
    prepare: dict[str, Any]
    ts: float

    def to_json(self) -> str:
        payload: dict[str, Any] = {"schema": SCHEMA, **asdict(self)}
        payload["read_refs"] = list(self.read_refs)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Row:
        raw = json.loads(line)
        raw.pop("schema", None)
        raw["read_refs"] = tuple(raw.get("read_refs", ()))
        return cls(**raw)


class Journal:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        run_id: str,
        config: str,
        engine: str,
        signal: str,
        doc: str,
        rendered: RenderedItem,
        answer: EngineAnswer,
        prepare: PrepareCost,
        contended: bool,
    ) -> Row:
        row = Row(
            run_id=run_id,
            config=config,
            engine=engine,
            signal=signal,
            doc=doc,
            item_id=rendered.id,
            rotation=rendered.rotation,
            gold=rendered.gold,
            text=answer.text,
            abstained=answer.abstained,
            read_refs=answer.read_refs,
            llm_calls=answer.llm_calls,
            prompt_chars=answer.prompt_chars,
            wall_s=answer.wall_s,
            contended=contended,
            error=answer.error,
            grounded=answer.grounded,
            options=dict(rendered.options),
            kind=str(rendered.item.kind),
            trace=answer.trace,
            prepare={
                "llm_calls": prepare.llm_calls,
                "prompt_chars": prepare.prompt_chars,
                "wall_s": prepare.wall_s,
                "cached": prepare.cached,
            },
            ts=time.time(),
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(row.to_json() + "\n")
        return row

    def read(self) -> Iterator[Row]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield Row.from_json(line)

    def latest_run_per_engine(self) -> dict[str, str]:
        seen: dict[str, tuple[float, str]] = {}
        for row in self.read():
            best = seen.get(row.engine)
            if best is None or row.ts > best[0]:
                seen[row.engine] = (row.ts, row.run_id)
        return {engine: run for engine, (_, run) in seen.items()}
