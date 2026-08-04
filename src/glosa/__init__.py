"""glosa — a grounded document-reading engine for Docling Studio.

Layered hexagonally: `domain` (pure logic) depends only on `ports`
(protocols); `infra` and `adapters` implement them. This module re-exports the
public surface so callers need not know the layout.
"""

from glosa.adapters.legacy import LegacyIteration, LegacyResult, to_legacy
from glosa.adapters.studio import GlosaReasoningRunner
from glosa.domain.budget import Budget
from glosa.domain.errors import (
    BackendError,
    BudgetExhausted,
    DocumentParseError,
    GlosaError,
    ReasoningParseError,
)
from glosa.domain.hybrid import HybridConfig, HybridStrategy
from glosa.domain.index import DocIndex, Unit
from glosa.domain.rank import Candidate, UnitRanker
from glosa.domain.values import (
    Element,
    Excerpt,
    RunStatus,
    Scope,
    Span,
    Step,
    Trace,
    UnitKind,
)
from glosa.infra.docling.projection import (
    DoclingProjection,
    DoclingProjector,
    node_id_for,
    page_node_id,
)
from glosa.infra.llm.ollama import OllamaChatModel
from glosa.infra.llm.openai import OpenAIChatModel, StructuredMode
from glosa.ports.chat import ChatModel, Message
from glosa.ports.document import DocumentProjection, DocumentProjector, TreeReader

__version__ = "0.1.0.dev0"

__all__ = [
    "BackendError",
    "Budget",
    "BudgetExhausted",
    "Candidate",
    "ChatModel",
    "DocIndex",
    "DoclingProjection",
    "DoclingProjector",
    "DocumentParseError",
    "DocumentProjection",
    "DocumentProjector",
    "Element",
    "Excerpt",
    "GlosaError",
    "GlosaReasoningRunner",
    "HybridConfig",
    "HybridStrategy",
    "LegacyIteration",
    "LegacyResult",
    "Message",
    "OllamaChatModel",
    "OpenAIChatModel",
    "ReasoningParseError",
    "RunStatus",
    "Scope",
    "Span",
    "Step",
    "StructuredMode",
    "Trace",
    "TreeReader",
    "Unit",
    "UnitKind",
    "UnitRanker",
    "__version__",
    "node_id_for",
    "page_node_id",
    "to_legacy",
]
