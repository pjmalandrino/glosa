"""Exception hierarchy.

`ReasoningParseError` deliberately mirrors the constructor signature of Docling
Studio's `domain.ports.ReasoningParseError` so an integrator can either let it
propagate through a translating adapter or hand its own class to
`GlosaReasoningRunner(parse_error_factory=...)`.
"""

from __future__ import annotations


class GlosaError(Exception):
    """Base class for every error raised by glosa."""


class DocumentParseError(GlosaError):
    """The supplied payload is not a valid serialized `DoclingDocument`."""


class ReasoningParseError(GlosaError):
    """The model could not produce a parseable structured answer.

    Raised only after constrained decoding, a repair round-trip and the
    configured retries have all failed — i.e. the backend is genuinely unable
    to satisfy the schema, not a transient formatting slip.
    """

    def __init__(self, model_id: str, reason: str = "no parseable answer") -> None:
        super().__init__(f"{model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


class BudgetExhausted(GlosaError):
    """A step, LLM-call or wall-clock budget ran out mid-run.

    Callers should not normally see this: the strategy catches it and returns a
    partial `Trace` with `status=BUDGET_EXHAUSTED` instead of failing the run.
    """


class BackendError(GlosaError):
    """The LLM backend was unreachable or returned an unusable HTTP response.

    `status_code` is the HTTP status when there was one; `retryable` says
    whether the failure was transient (a 503, a dropped connection) or
    deterministic (a 401, a malformed body) — so a host can tell back-pressure
    from misconfiguration without parsing the message.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
