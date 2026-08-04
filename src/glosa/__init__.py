"""glosa — a grounded document-reading engine for `DoclingDocument`."""

from glosa.adapters.legacy import to_legacy
from glosa.adapters.studio import GlosaReasoningRunner
from glosa.document.index import DocIndex, Unit
from glosa.errors import (
    BackendError,
    BudgetExhausted,
    DocumentParseError,
    GlosaError,
    ReasoningParseError,
)
from glosa.llm.ollama import OllamaChatModel
from glosa.llm.openai import OpenAIChatModel, StructuredMode
from glosa.llm.port import ChatModel, Message
from glosa.strategy.navigate import NavigateConfig, NavigateStrategy
from glosa.studio.ports import TreeReader
from glosa.studio.projection import Element, Scope, StudioProjection, node_id_for, page_node_id
from glosa.studio.tree import BundledTreeReader
from glosa.types import (
    Excerpt,
    LegacyIteration,
    LegacyResult,
    RunStatus,
    Span,
    Step,
    Trace,
    UnitKind,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BackendError",
    "BudgetExhausted",
    "BundledTreeReader",
    "ChatModel",
    "DocIndex",
    "DocumentParseError",
    "Element",
    "Excerpt",
    "GlosaError",
    "GlosaReasoningRunner",
    "LegacyIteration",
    "LegacyResult",
    "Message",
    "NavigateConfig",
    "NavigateStrategy",
    "OllamaChatModel",
    "OpenAIChatModel",
    "ReasoningParseError",
    "RunStatus",
    "Scope",
    "Span",
    "Step",
    "StructuredMode",
    "StudioProjection",
    "Trace",
    "TreeReader",
    "Unit",
    "UnitKind",
    "__version__",
    "node_id_for",
    "page_node_id",
    "to_legacy",
]
