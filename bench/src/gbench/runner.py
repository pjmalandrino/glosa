"""The application service: run engines over items, write rows.

Depends on ports and domain only — it never learns what an engine is made of,
which is what lets the controls, glosa, and two competitors that are not
installed go through the same code path.

It does exactly one thing worth explaining. Items run concurrently, because
1 920 executions served serially by a local 8B is most of a day — and
concurrency **destroys wall-clock as a comparable metric**, since the engines
then queue behind each other on one GPU. So every row carries `contended`, the
report refuses to rank on contended timings, and `--latency-pass` re-runs a
fixed subset at concurrency 1 for the p50/p95 numbers. An engine whose claim is
"two round-trips instead of ten sequential" must not get to prove it with a
contended stopwatch.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gbench.domain.item import flip_subsample, schedule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gbench.domain.item import Item
    from gbench.infra.journal import Journal, Row
    from gbench.ports.engine import BenchDocument, Engine, PrepareCost


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What a single `gbench run` invocation will do.

    The default is not "four rotations of everything". At 200 items and five
    engines that is 4 000 executions, and most of them buy nothing: what four
    rotations of *every* item guards against is aggregate position bias, and a
    suite of 200 gets that for free by giving each item a deterministic
    assigned rotation (`item.assigned_rotation`). What it does not get for free
    is per-item stability, so `flip_sample` items are swept in full.

    200 items becomes 320 executions per engine instead of 800, and both
    numbers rotation exists to produce still come out.
    """

    sweep: bool = False
    """Four rotations of every item. Right for a 60-item suite, wasteful here."""
    flip_sample: int = 40
    """Items swept in full — stratified over kinds, stable between runs."""
    concurrency: int = 4
    latency_pass: bool = False
    """Serial, on a subset, for timings only."""
    latency_items: int = 12

    @property
    def effective_concurrency(self) -> int:
        return 1 if self.latency_pass else max(1, self.concurrency)


async def run(
    *,
    engines: Sequence[Engine],
    documents: dict[str, BenchDocument],
    items: Sequence[Item],
    journal: Journal,
    config: str,
    plan: RunPlan | None = None,
    run_id: str | None = None,
) -> list[Row]:
    """Execute the plan, appending one row per (item, engine, rotation)."""
    plan = plan or RunPlan()
    run_id = run_id or uuid.uuid4().hex[:12]
    selected = list(items[: plan.latency_items] if plan.latency_pass else items)

    rows: list[Row] = []
    for engine in engines:
        costs: dict[str, PrepareCost] = {}
        for slug, document in documents.items():
            costs[slug] = await engine.prepare(document)

        gate = asyncio.Semaphore(plan.effective_concurrency)
        contended = plan.effective_concurrency > 1
        swept = frozenset(flip_subsample(selected, plan.flip_sample))
        produced = await asyncio.gather(
            *(
                _one_item(
                    engine=engine,
                    document=documents[item.doc],
                    item=item,
                    plan=plan,
                    swept=swept,
                    gate=gate,
                    journal=journal,
                    run_id=run_id,
                    config=config,
                    prepare=costs[item.doc],
                    contended=contended,
                )
                for item in selected
            )
        )
        for produced_rows in produced:
            rows.extend(produced_rows)

    return rows


async def _one_item(
    *,
    engine: Engine,
    document: BenchDocument,
    item: Item,
    plan: RunPlan,
    swept: frozenset[str],
    gate: asyncio.Semaphore,
    journal: Journal,
    run_id: str,
    config: str,
    prepare: PrepareCost,
    contended: bool,
) -> list[Row]:
    """All rotations of one item, against one engine.

    A module-level function rather than a closure: everything it needs is
    passed, so nothing can be captured from a loop that has since moved on.
    """
    rows: list[Row] = []
    for rendered in schedule([item], sweep=plan.sweep or item.id in swept):
        async with gate:
            answer = await engine.answer(document, rendered)
        rows.append(
            journal.append(
                run_id=run_id,
                config=config,
                engine=engine.name,
                signal=engine.capabilities.abstention_signal,
                doc=item.doc,
                rendered=rendered,
                answer=answer,
                prepare=prepare,
                contended=contended,
            )
        )
    return rows
