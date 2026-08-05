"""PageIndex as a bench engine.  **B2 — not yet run.**

PageIndex is two halves, and only the first ships as a program:

1. **the index** — `run_pageindex.py --md_path doc.md` walks the document and
   pays an LLM call per node to write a title and a summary, emitting a tree of
   `{title, node_id, start_index, end_index, summary, nodes[]}`. Run as a
   subprocess, once per document, cached by document hash;
2. **the tree search** — descend the tree by showing the model a level's titles
   and summaries, pick a node, repeat to a leaf, then answer from its text. The
   OSS repo documents this as a recipe rather than shipping it as an API, so
   the harness implements it here, in about sixty lines, and **says so**: a
   number produced by this adapter measures PageIndex's *index* plus the
   harness's faithful-but-not-upstream traversal.

That caveat is the honest cost of including PageIndex at all, and it is why
this file carries the traversal prompt in full: a reader who thinks it is
unfaithful can see exactly what was asked.

The index is where PageIndex's whole bet sits. It is the best map of the three
— a written summary per node, where `docling-agent` has headings and glosa has
each section's own opening line — and it is paid for up front, per document,
again on every new version of it. `prepare()` exists to make that visible
(`EVAL.md` §4.3): PageIndex's cold column is large and its warm column is
competitive, and collapsing the two into one number would hide the entire
trade-off.

Behind the `[pageindex]` extra. Points LiteLLM at the same Ollama endpoint as
everyone else via the `ollama/<model>` model string.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glosa.ports.chat import system, user

from gbench.ports.engine import BenchDocument, Capabilities, EngineAnswer, PrepareCost

if TYPE_CHECKING:
    from glosa.ports.chat import ChatModel

    from gbench.domain.item import RenderedItem

DESCEND = """You are navigating a document by its table of contents.

Question:
{question}

Nodes at this level:
{nodes}

Reply with the node_id of the single node most likely to contain the answer, \
and nothing else."""

ANSWER = """Document extract:

{context}

---

{prompt}"""

MAX_DEPTH = 4


class PageIndexEngine:
    """Index built by upstream's script; traversal implemented here."""

    def __init__(
        self,
        model: ChatModel,
        *,
        script: Path,
        cache_dir: Path,
        model_string: str,
        max_depth: int = MAX_DEPTH,
    ) -> None:
        self._model = model
        self._script = script
        self._cache = cache_dir
        self._model_string = model_string
        self._max_depth = max_depth
        self._trees: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "pageindex"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            read_refs=True,
            abstention=False,
            llm_calls=True,
            groundedness=False,
            abstention_signal="none — the engine has no such signal",
        )

    async def prepare(self, doc: BenchDocument) -> PrepareCost:
        """Build (or reuse) the tree. This is where PageIndex spends."""
        target = self._cache / f"{doc.doc_hash or doc.slug}.tree.json"
        started = time.monotonic()

        if target.exists():
            self._trees[doc.slug] = json.loads(target.read_text(encoding="utf-8"))
            return PrepareCost(llm_calls=0, prompt_chars=0, cached=True)

        markdown = self._cache / f"{doc.slug}.md"
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(doc.markdown, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            "python3",
            str(self._script),
            "--md_path",
            str(markdown),
            "--model",
            self._model_string,
            "--if-add-node-summary",
            "yes",
            "--if-add-node-id",
            "yes",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"pageindex failed ({process.returncode}): {stderr.decode()[-800:]}")

        tree = json.loads(stdout.decode() or target.read_text(encoding="utf-8"))
        target.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
        self._trees[doc.slug] = tree
        return PrepareCost(
            llm_calls=_count_nodes(tree),  # one summary call per node, upstream's design
            prompt_chars=len(doc.markdown),
            wall_s=round(time.monotonic() - started, 3),
        )

    async def answer(self, doc: BenchDocument, item: RenderedItem) -> EngineAnswer:
        tree = self._trees.get(doc.slug)
        if tree is None:
            await self.prepare(doc)
            tree = self._trees[doc.slug]

        started = time.monotonic()
        calls = 0
        visited: list[str] = []
        nodes = _children(tree)

        try:
            for _ in range(self._max_depth):
                if not nodes:
                    break
                listing = "\n".join(
                    f"- {node.get('node_id')}: {node.get('title', '')} — {node.get('summary', '')}"
                    for node in nodes
                )
                calls += 1
                picked = await self._model.complete(
                    [
                        system("Reply with a node_id and nothing else."),
                        user(DESCEND.format(question=item.item.question, nodes=listing)),
                    ],
                    max_tokens=32,
                )
                node = _match(nodes, picked) or nodes[0]
                visited.append(str(node.get("node_id", "")))
                children = _children(node)
                if not children:
                    break
                nodes = children

            context = _text_of(doc.markdown, node) if visited else doc.markdown[:8_000]
            calls += 1
            text = await self._model.complete(
                [
                    system("Answer with a single letter and nothing else."),
                    user(ANSWER.format(context=context, prompt=item.prompt)),
                ],
                max_tokens=8,
            )
        except Exception as exc:
            return EngineAnswer(
                text="",
                error=f"{type(exc).__name__}: {exc}",
                llm_calls=calls,
                wall_s=round(time.monotonic() - started, 3),
            )

        return EngineAnswer(
            text=text,
            abstained=False,  # no such signal — the finding, not a handicap
            read_refs=tuple(visited),
            llm_calls=calls,
            prompt_chars=len(context),
            wall_s=round(time.monotonic() - started, 3),
            trace={"path": visited},
        )

    async def aclose(self) -> None:
        await self._model.aclose()


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("nodes") or node.get("structure") or []
    return [child for child in children if isinstance(child, dict)]


def _match(nodes: list[dict[str, Any]], reply: str) -> dict[str, Any] | None:
    cleaned = reply.strip().strip("\"'` ")
    for node in nodes:
        if str(node.get("node_id", "")) and str(node["node_id"]) in cleaned:
            return node
    return None


def _text_of(markdown: str, node: dict[str, Any]) -> str:
    """Upstream indexes markdown by line range; slice the same source back out."""
    lines = markdown.splitlines()
    start = int(node.get("start_index", 0) or 0)
    end = int(node.get("end_index", len(lines)) or len(lines))
    return "\n".join(lines[start : end + 1])[:8_000]


def _count_nodes(tree: dict[str, Any]) -> int:
    return 1 + sum(_count_nodes(child) for child in _children(tree))
