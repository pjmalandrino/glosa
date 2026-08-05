"""Composition root.

    gbench lint   --corpus corpus                       # no model, no GPU
    gbench run    --corpus corpus --engines glosa,closed-book,oracle-context
    gbench score  --journal runs/journal.jsonl --corpus corpus [--policy strict]

`lint` and `score` never touch a model — the first keeps the corpus honest, the
second turns an existing journal into the table. Only `run` needs a backend, and
only the engines named on the command line are imported, which is why a machine
with neither `mellea` nor `litellm` installed can still run and score glosa
against both controls.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from glosa.infra.llm.ollama import DEFAULT_HOST, OllamaChatModel
from glosa.infra.llm.openai import DEFAULT_BASE_URL, OpenAIChatModel
from glosa.ports.chat import ChatModel

from gbench.domain import report
from gbench.domain.lint import failed, lint
from gbench.domain.scoring import Policy
from gbench.infra.corpus import FileCorpus
from gbench.infra.journal import Journal, config_hash
from gbench.runner import RunPlan, run
from gbench.scoreboard import build

ENGINE_NAMES = ("glosa", "closed-book", "oracle-context", "docling-agent", "pageindex")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("lint", help="check the corpus before anybody quotes it")
    check.add_argument("--corpus", type=Path, default=Path("corpus"))

    execute = sub.add_parser("run", help="run engines over the suite, append to the journal")
    execute.add_argument("--corpus", type=Path, default=Path("corpus"))
    execute.add_argument("--journal", type=Path, default=Path("runs/journal.jsonl"))
    execute.add_argument("--engines", default="glosa,closed-book,oracle-context")
    execute.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    execute.add_argument("--base-url", default=None)
    execute.add_argument("--model", default="granite3.3:8b")
    execute.add_argument("--api-key", default=None)
    execute.add_argument("--rotations", type=int, default=4)
    execute.add_argument("--concurrency", type=int, default=4)
    execute.add_argument("--doc", default=None, help="restrict to one document")
    execute.add_argument(
        "--latency-pass",
        action="store_true",
        help="serial run over a subset — the only timings that may be quoted",
    )
    execute.add_argument("--profile", choices=("full", "smoke"), default="full")

    table = sub.add_parser("score", help="turn a journal into the table")
    table.add_argument("--corpus", type=Path, default=Path("corpus"))
    table.add_argument("--journal", type=Path, default=Path("runs/journal.jsonl"))
    table.add_argument("--policy", choices=("declared", "strict"), default="declared")
    table.add_argument("--keep-leaky", action="store_true")
    table.add_argument("--out", type=Path, default=None)

    return parser


def _model(args: argparse.Namespace) -> ChatModel:
    if args.provider == "ollama":
        return OllamaChatModel(base_url=args.base_url or DEFAULT_HOST, model_id=args.model)
    return OpenAIChatModel(
        base_url=args.base_url or DEFAULT_BASE_URL, model_id=args.model, api_key=args.api_key
    )


def _engines(names: list[str], args: argparse.Namespace, corpus: FileCorpus) -> list[object]:
    """Import an adapter only when it is asked for — that is the whole point of
    the extras in `pyproject.toml`."""
    built: list[object] = []
    for name in names:
        if name == "glosa":
            from gbench.infra.engines.glosa_engine import GlosaEngine

            built.append(GlosaEngine(_model(args)))
        elif name == "closed-book":
            from gbench.infra.engines.controls import ClosedBookEngine

            built.append(ClosedBookEngine(_model(args)))
        elif name == "oracle-context":
            from gbench.infra.engines.controls import OracleContextEngine

            built.append(OracleContextEngine(_model(args), corpus.text_of))
        elif name == "docling-agent":
            from gbench.infra.engines.docling_agent import DoclingAgentEngine

            built.append(
                DoclingAgentEngine(base_url=args.base_url or DEFAULT_HOST, model_id=args.model)
            )
        elif name == "pageindex":
            from gbench.infra.engines.pageindex import PageIndexEngine

            built.append(
                PageIndexEngine(
                    _model(args),
                    script=Path(corpus.suite["pageindex"]["script"]),
                    cache_dir=Path(corpus.suite["pageindex"]["cache"]),
                    model_string=f"ollama/{args.model}",
                )
            )
        else:
            raise SystemExit(f"unknown engine {name!r}; known: {', '.join(ENGINE_NAMES)}")
    return built


def _lint(args: argparse.Namespace) -> int:
    corpus = FileCorpus(args.corpus)
    findings = lint(corpus.items(), text_of=corpus.text_of, doc_text=corpus.doc_text)
    for finding in findings:
        print(finding)
    print(f"\n{len(corpus.items())} item(s), {len(findings)} finding(s)")
    return 1 if failed(findings) else 0


def _run(args: argparse.Namespace) -> int:
    corpus = FileCorpus(args.corpus)
    names = [name.strip() for name in args.engines.split(",") if name.strip()]
    engines = _engines(names, args, corpus)

    items = list(corpus.items(args.doc))
    plan = RunPlan(
        rotations=1 if args.profile == "smoke" else args.rotations,
        concurrency=args.concurrency,
        latency_pass=args.latency_pass,
    )
    if args.profile == "smoke":
        items = items[:12]

    documents = {slug: corpus.document(slug) for slug in {item.doc for item in items}}
    config = config_hash(
        {
            "model": args.model,
            "provider": args.provider,
            "rotations": plan.rotations,
            "docs": sorted(documents),
            "items": sorted(item.id for item in items),
        }
    )

    async def go() -> int:
        try:
            rows = await run(
                engines=engines,  # type: ignore[arg-type]
                documents=documents,
                items=items,
                journal=Journal(args.journal),
                config=config,
                plan=plan,
            )
        finally:
            for engine in engines:
                await engine.aclose()  # type: ignore[attr-defined]
        print(f"{len(rows)} row(s) appended to {args.journal} (config {config})")
        return 0

    return asyncio.run(go())


def _score(args: argparse.Namespace) -> int:
    corpus = FileCorpus(args.corpus)
    journal = Journal(args.journal)
    board = build(
        journal.read(),
        corpus.items(),
        policy=Policy(args.policy),
        exclude_leaky=not args.keep_leaky,
    )
    lines = [
        f"# Bench — policy `{board.policy}`",
        "",
        report.headline(board.rows),
        "",
        "## Paired comparisons",
        "",
        report.comparisons(board.deltas),
        "",
        "## What each engine can report",
        "",
        report.capabilities({row.engine: row.abstention_signal for row in board.rows}),
    ]
    if board.leaky:
        lines += [
            "",
            f"## Leaky items ({len(board.leaky)}) — answered closed-book, excluded",
            "",
            ", ".join(board.leaky),
        ]
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lint":
        return _lint(args)
    if args.command == "run":
        return _run(args)
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
