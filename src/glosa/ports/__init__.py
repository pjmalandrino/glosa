"""Ports — the interfaces the domain needs, and the ones adapters consume.

Protocols only. A port may reference domain values; it never references an
implementation.
"""

from glosa.ports.chat import ChatModel, Message, system, user
from glosa.ports.document import DocumentProjection, DocumentProjector, TreeReader

__all__ = [
    "ChatModel",
    "DocumentProjection",
    "DocumentProjector",
    "Message",
    "TreeReader",
    "system",
    "user",
]
