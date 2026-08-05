"""Composition root.

    gbench manifest --sets cs,math,q-bio --since 2026-02-01   # choose the papers
    gbench fetch                                              # pull the PDFs, check the bytes
    gbench convert                                       # Docling → docling.json, sections.json
    gbench lint                                               # no model, no GPU
    gbench run    --engines glosa,closed-book,abstract-only,oracle-context
    gbench score  [--policy strict]

Four of the six never touch a model. `manifest` and `fetch` need the network,
`convert` needs Docling, and `lint`/`score` need neither — the first keeps the
corpus honest, the second turns an existing journal into the table. Only `run`
needs a backend, and only the engines named on the command line are imported,
which is why a machine with neither `mellea` nor `litellm` installed can still
run and score glosa against all three controls.
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
from gbench.domain.lint import LicencePolicy, failed, lint
from gbench.domain.scoring import Policy
from gbench.infra.corpus import FileCorpus
from gbench.infra.journal import Journal, config_hash
from gbench.runner import RunPlan, run
from gbench.scoreboard import build

ENGINE_NAMES = (
    "glosa",
    "closed-book",
    "abstract-only",
    "oracle-context",
    "docling-agent",
    "pageindex",
)
CONTROLS = "closed-book,abstract-only,oracle-context"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("lint", help="check the corpus before anybody quotes it")
    check.add_argument("--corpus", type=Path, default=Path("corpus"))
    check.add_argument(
        "--partial",
        action="store_true",
        help="do not require the five-kinds-per-paper quota (corpus still being authored)",
    )
    check.add_argument(
        "--verify-quotes",
        action="store_true",
        help="fail if the papers are not converted — the offline check cannot run without them",
    )

    execute = sub.add_parser("run", help="run engines over the suite, append to the journal")
    execute.add_argument("--corpus", type=Path, default=Path("corpus"))
    execute.add_argument("--journal", type=Path, default=Path("runs/journal.jsonl"))
    execute.add_argument("--engines", default=f"glosa,{CONTROLS}")
    execute.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    execute.add_argument("--base-url", default=None)
    execute.add_argument("--model", default="granite3.3:8b")
    execute.add_argument("--api-key", default=None)
    execute.add_argument(
        "--sweep",
        action="store_true",
        help="four rotations of every item (default: assigned rotation + a swept sample)",
    )
    execute.add_argument("--flip-sample", type=int, default=40)
    execute.add_argument("--concurrency", type=int, default=4)
    execute.add_argument("--doc", default=None, help="restrict to one document")
    execute.add_argument(
        "--latency-pass",
        action="store_true",
        help="serial run over a subset — the only timings that may be quoted",
    )
    execute.add_argument("--profile", choices=("full", "smoke"), default="full")

    choose = sub.add_parser("manifest", help="build the manifest, by hand or by harvest")
    choose.add_argument("--corpus", type=Path, default=Path("corpus"))
    choose.add_argument(
        "--from-list",
        type=Path,
        default=None,
        help="a pasted list of arXiv ids or URLs, one per line, optional category after each "
        "— the hand-picked path, and the one that needs no licence hunting",
    )
    choose.add_argument("--sets", default="cs,math,q-bio,physics:cond-mat")
    choose.add_argument("--since", default=None, help="YYYY-MM-DD — after the model's cutoff")
    choose.add_argument("--until", default=None)
    choose.add_argument("--per-set", type=int, default=5)
    choose.add_argument(
        "--policy",
        choices=("fetch-only", "redistributable"),
        default="fetch-only",
        help="fetch-only commits hashes and accepts any licence; redistributable commits "
        "the text and accepts only CC-BY or better",
    )
    choose.add_argument("--out", type=Path, default=None, help="default: <corpus>/manifest.yaml")

    pull = sub.add_parser("fetch", help="download the PDFs the manifest pins")
    pull.add_argument("--corpus", type=Path, default=Path("corpus"))
    pull.add_argument("--doc", default=None)

    convert = sub.add_parser("convert", help="PDF → docling.json + doc.md + refs.json")
    convert.add_argument("--corpus", type=Path, default=Path("corpus"))
    convert.add_argument("--doc", default=None)
    convert.add_argument(
        "--with-text",
        action="store_true",
        help="also write sections.json — only commit it under `redistributable`",
    )

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
        elif name == "abstract-only":
            from gbench.infra.engines.controls import AbstractOnlyEngine

            built.append(AbstractOnlyEngine(_model(args), corpus.abstract_text))
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
    unreadable = [slug for slug in corpus.slugs if not corpus.has_text(slug)]
    if args.verify_quotes and unreadable:
        print(
            f"--verify-quotes needs the papers: {len(unreadable)} of {len(corpus.slugs)} "
            f"not converted. Run `gbench fetch && gbench convert`."
        )
        return 2

    findings = lint(
        corpus.items(),
        text_of=corpus.text_of,
        doc_text=corpus.doc_text,
        abstract_refs=corpus.abstract_refs,
        licenses=corpus.licenses(),
        policy=corpus.policy,
        has_text=corpus.has_text,
        require_quota=not args.partial,
    )
    for finding in findings:
        print(finding)
    print(f"\n{len(corpus.items())} item(s), {len(findings)} finding(s) — policy {corpus.policy}")
    if unreadable and not args.verify_quotes:
        print(
            f"{len(unreadable)} paper(s) not converted: quotes unverified. "
            f"`gbench fetch && gbench convert && gbench lint --verify-quotes`"
        )
    return 1 if failed(findings) else 0


def _run(args: argparse.Namespace) -> int:
    corpus = FileCorpus(args.corpus)
    names = [name.strip() for name in args.engines.split(",") if name.strip()]
    engines = _engines(names, args, corpus)

    items = list(corpus.items(args.doc))
    plan = RunPlan(
        sweep=args.sweep,
        flip_sample=0 if args.profile == "smoke" else args.flip_sample,
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
            "sweep": plan.sweep,
            "flip_sample": plan.flip_sample,
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
        "",
        "## Breakdowns",
        "",
        report.breakdown("By kind", board.by_kind),
        "",
        report.breakdown("By domain", board.by_domain),
        "",
        "Items behind each column: "
        + ", ".join(f"{key} {value}" for key, value in sorted(board.counts.items())),
    ]
    if board.leaky:
        lines += [
            "",
            f"## Leaky items ({len(board.leaky)}) — answered with no navigation, excluded",
            "",
            ", ".join(board.leaky),
        ]
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _manifest(args: argparse.Namespace) -> int:
    """Write the corpus's provenance file. The only command that picks papers.

    Two ways in, and the hand-picked one is not the fallback. Which fields a
    corpus should span is a judgement call — a maths paper and an econometrics
    paper fail a reader differently, and arXiv's firehose has no opinion about
    that. `--from-list` takes whatever a person pasted; harvesting is for when
    nobody has an opinion yet.
    """
    import yaml

    from gbench.infra.arxiv import parse_list, propose

    if args.from_list:
        candidates = parse_list(args.from_list.read_text(encoding="utf-8"))
        source = f"{len(candidates)} paper(s) from {args.from_list}"
    else:
        if not (args.since and args.until):
            raise SystemExit("harvesting needs --since and --until; or pass --from-list")
        sets = [name.strip() for name in args.sets.split(",") if name.strip()]
        candidates = propose(sets=sets, since=args.since, until=args.until, per_set=args.per_set)
        source = f"{len(candidates)} paper(s) across {len(sets)} set(s)"

    payload = {
        "licence_policy": args.policy,
        "documents": [
            {
                "slug": candidate.slug,
                "title": candidate.title,
                "arxiv_id": candidate.arxiv_id,
                "version": candidate.version,
                "license": candidate.license,
                "primary_category": candidate.primary_category,
                "sha256": "",  # filled by `gbench fetch`
            }
            for candidate in candidates
        ],
    }
    target = args.out or (args.corpus / "manifest.yaml")
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), "utf-8")
    print(f"{source} → {target}")
    print("sha256 and licence are empty until `gbench fetch` fills them.")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    import yaml

    from gbench.infra.arxiv import fetch, licence_of

    corpus = FileCorpus(args.corpus)
    manifest = args.corpus / "manifest.yaml"
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    changed = False

    for entry in payload.get("documents", []):
        slug = str(entry["slug"])
        if args.doc and slug != args.doc:
            continue
        if entry.get("generated"):
            continue
        paper = corpus.papers[slug]
        target = corpus.dir_of(slug) / "source.pdf"
        digest = fetch(paper.pdf_url, target, expected_sha256=paper.sha256)
        if not paper.sha256:
            entry["sha256"] = digest
            changed = True
            print(f"{slug}: pinned {digest[:16]}…")
        else:
            print(f"{slug}: verified")
        if not paper.license and paper.arxiv_id:
            # Best effort, and only ever additive: under `fetch-only` a licence
            # we could not read is a warning, not a rejection.
            found = licence_of(paper.arxiv_id)
            if found:
                entry["license"] = found
                changed = True

    if changed:
        manifest.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), "utf-8")
        print(f"{manifest} updated with the pinned digests")
    return 0


def _convert(args: argparse.Namespace) -> int:
    """PDF → the artefacts the engines and the linter need.

    Docling is a heavy dependency with model weights behind it, so it lives in
    the `[convert]` extra and is imported here rather than at module load. The
    output is committed only in part: `sections.json` goes into git, the rest
    is reproducible from the manifest.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError:
        print("convert needs `uv sync --extra convert` (Docling and its models)")
        return 2

    from gbench.infra.corpus import Section, write_refs, write_sections

    corpus = FileCorpus(args.corpus)
    converter = DocumentConverter()
    for slug in corpus.slugs:
        if args.doc and slug != args.doc:
            continue
        pdf = corpus.dir_of(slug) / "source.pdf"
        if not pdf.exists():
            print(f"{slug}: no source.pdf — run `gbench fetch` first")
            continue
        document = converter.convert(str(pdf)).document
        (corpus.dir_of(slug) / "docling.json").write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
        (corpus.dir_of(slug) / "doc.md").write_text(document.export_to_markdown(), encoding="utf-8")
        sections: list[Section] = list(corpus.project(slug))
        write_refs(corpus.dir_of(slug) / "refs.json", slug, sections)
        if args.with_text or corpus.policy is LicencePolicy.REDISTRIBUTABLE:
            write_sections(corpus.dir_of(slug) / "sections.json", slug, sections)
        print(f"{slug}: {len(sections)} element(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lint":
        return _lint(args)
    if args.command == "manifest":
        return _manifest(args)
    if args.command == "fetch":
        return _fetch(args)
    if args.command == "convert":
        return _convert(args)
    if args.command == "run":
        return _run(args)
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
