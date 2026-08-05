"""Command line entry point.

    glosa ask  --document analysis.json --query "What is the penalty clause?"
    glosa map  --document analysis.json

`ask` prints the answer and the trace; `--json` prints the whole trace as JSON,
which is what a replay harness will consume.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from glosa.domain.hybrid import HybridConfig, HybridStrategy
from glosa.domain.index import DocIndex
from glosa.domain.outline import render_outline
from glosa.domain.values import Trace
from glosa.infra.docling.projection import DoclingProjector
from glosa.infra.llm.ollama import DEFAULT_HOST, OllamaChatModel
from glosa.infra.llm.openai import DEFAULT_BASE_URL, OpenAIChatModel
from glosa.ports.chat import ChatModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glosa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question against a document")
    ask.add_argument("--document", required=True, type=Path, help="DoclingDocument JSON file")
    ask.add_argument("--query", required=True)
    ask.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    ask.add_argument("--base-url", default=None)
    ask.add_argument("--model", default="granite3.3:8b")
    ask.add_argument("--api-key", default=None)
    ask.add_argument("--max-steps", type=int, default=HybridConfig().max_steps)
    ask.add_argument("--timeout", type=float, default=180.0)
    ask.add_argument("--json", action="store_true", help="print the full trace as JSON")

    show = sub.add_parser("map", help="print the document map glosa navigates by")
    show.add_argument("--document", required=True, type=Path)
    show.add_argument("--budget", type=int, default=6_000)

    return parser


def _index(path: Path) -> DocIndex:
    """Composition: pick a projector, hand the domain a projection."""
    projection = DoclingProjector().project(path.read_text(encoding="utf-8"))
    return DocIndex(projection)


def _model_for(args: argparse.Namespace) -> ChatModel:
    if args.provider == "ollama":
        return OllamaChatModel(
            base_url=args.base_url or DEFAULT_HOST,
            model_id=args.model,
            timeout=args.timeout,
        )
    return OpenAIChatModel(
        base_url=args.base_url or DEFAULT_BASE_URL,
        model_id=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
    )


async def _ask(args: argparse.Namespace) -> int:
    index = _index(args.document)
    model = _model_for(args)
    strategy = HybridStrategy(
        model, HybridConfig(max_steps=args.max_steps, deadline_s=args.timeout)
    )
    try:
        trace = await strategy.run(index, args.query)
    finally:
        await model.aclose()

    if args.json:
        print(json.dumps(_as_dict(trace), indent=2, ensure_ascii=False))
        return 0

    print(f"status   : {trace.status}")
    print(f"model    : {trace.model_id}")
    print(f"calls    : {trace.llm_calls}   elapsed: {trace.elapsed_s}s")
    print()
    for step in trace.steps:
        flag = "✓" if step.sufficient else "·"
        extra = " (fallback)" if step.fallback else ""
        print(f"  {flag} {step.index}. {step.ref} — {step.title}{extra}")
        print(f"      why : {step.reason}")
        print(f"      read: {step.excerpt_chars} chars, pages {list(step.pages) or '—'}")
        if step.grounded is False:
            # An answer whose citation is not in the text it read is the one
            # thing a reader must not scroll past.
            print(f"      /!\\  cited a sentence absent from this section: {step.quote[:80]!r}")
        elif step.grounded:
            print(f"      cite: {step.quote[:80]}")
    print()
    print(trace.answer)
    return 0 if trace.converged else 1


def _show_map(args: argparse.Namespace) -> int:
    index = _index(args.document)
    print(f"{index.title} — {len(index.units)} unit(s) of kind '{index.kind}'")
    print(render_outline(index.units, char_budget=args.budget))
    return 0


def _as_dict(trace: Trace) -> dict[str, Any]:
    payload = dataclasses.asdict(trace)
    payload["status"] = str(trace.status)
    for step in payload["steps"]:
        step["kind"] = str(step["kind"])
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "map":
        return _show_map(args)
    return asyncio.run(_ask(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
