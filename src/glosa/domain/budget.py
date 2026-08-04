"""Run budgets.

A reading loop that cannot be bounded cannot be embedded in a request handler.
Steps, LLM calls and wall-clock are all capped, and running out is an ordinary
outcome (`RunStatus.BUDGET_EXHAUSTED` with a partial answer), not a crash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from glosa.domain.errors import BudgetExhausted


@dataclass(slots=True)
class Budget:
    """Mutable per-run budget tracker."""

    max_steps: int = 6
    max_llm_calls: int = 20
    deadline_s: float | None = 180.0
    calls: int = 0
    steps: int = 0
    _start: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.calls = 0
        self.steps = 0
        self._start = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining_s(self) -> float | None:
        if self.deadline_s is None:
            return None
        return max(0.0, self.deadline_s - self.elapsed_s)

    def spend_call(self) -> None:
        """Account for one LLM round-trip. Raises once the ceiling is reached."""
        if self.calls >= self.max_llm_calls:
            raise BudgetExhausted(f"llm call budget of {self.max_llm_calls} reached")
        self.calls += 1

    def spend_step(self) -> None:
        """Account for one read. Raises once the ceiling is reached."""
        if self.steps >= self.max_steps:
            raise BudgetExhausted(f"step budget of {self.max_steps} reached")
        self.steps += 1

    def check_deadline(self) -> None:
        if self.deadline_s is not None and self.elapsed_s >= self.deadline_s:
            raise BudgetExhausted(f"deadline of {self.deadline_s:.0f}s reached")

    def can_afford_extra_call(self) -> bool:
        """True when there is room for a non-essential call, e.g. composing a
        partial answer after the loop stopped."""
        if self.calls >= self.max_llm_calls:
            return False
        return self.remaining_s is None or self.remaining_s > 5.0
