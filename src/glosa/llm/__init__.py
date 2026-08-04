"""LLM backends. Everything reaches a model through `ChatModel`."""

from glosa.llm.ollama import OllamaChatModel
from glosa.llm.openai import OpenAIChatModel, StructuredMode
from glosa.llm.port import ChatModel, Message, system, user

__all__ = [
    "ChatModel",
    "Message",
    "OllamaChatModel",
    "OpenAIChatModel",
    "StructuredMode",
    "system",
    "user",
]
