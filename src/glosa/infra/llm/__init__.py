"""Chat backends implementing the `ChatModel` port."""

from glosa.infra.llm.ollama import OllamaChatModel
from glosa.infra.llm.openai import OpenAIChatModel, StructuredMode

__all__ = ["OllamaChatModel", "OpenAIChatModel", "StructuredMode"]
