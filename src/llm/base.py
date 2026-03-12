"""base.py - Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Contract every LLM provider must implement.

    A provider is responsible for constructing a LangChain ``BaseChatModel``
    from environment variables and the two inference knobs exposed in
    ``agent_config.yaml`` (``temperature``, ``top_p``).  All authentication
    details, model names, and endpoints are resolved inside ``build()``.
    """

    @abstractmethod
    def build(self, temperature: float, top_p: float) -> BaseChatModel:
        """Instantiate and return a ready-to-use LangChain chat model.

        Args:
            temperature: Sampling temperature (0-2).
            top_p: Nucleus-sampling probability mass (0-1).

        Returns:
            A ``BaseChatModel`` instance compatible with ``.bind_tools()``.
        """
