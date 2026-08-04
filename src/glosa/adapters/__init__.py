"""Driving adapters — how a host calls glosa."""

from glosa.adapters.legacy import LegacyIteration, LegacyResult, to_legacy
from glosa.adapters.studio import GlosaReasoningRunner

__all__ = ["GlosaReasoningRunner", "LegacyIteration", "LegacyResult", "to_legacy"]
