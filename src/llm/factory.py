"""factory.py - LLMFactory: selects and instantiates the right LLM provider.

Built-in providers
------------------
"azure"  → AzureOpenAIProvider  (requires langchain-openai)
"gemini" → GeminiProvider       (requires langchain-google-genai)
"ibm"    → IBMWatsonxProvider   (requires langchain-ibm)

Extending
---------
Register a custom provider at runtime before the first ``create()`` call::

    from src.llm.factory import LLMFactory
    from my_pkg import MyProvider

    LLMFactory.register("my-provider", MyProvider)
"""

from __future__ import annotations

from typing import Dict, Type

from langchain_core.language_models import BaseChatModel

from src.llm.azure import AzureOpenAIProvider
from src.llm.base import LLMProvider
from src.llm.gemini import GeminiProvider
from src.llm.ibm import IBMWatsonxProvider


class LLMFactory:
    _registry: Dict[str, Type[LLMProvider]] = {
        "azure": AzureOpenAIProvider,
        "gemini": GeminiProvider,
        "ibm": IBMWatsonxProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        """Register a new (or override an existing) provider by name."""
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, provider: str, temperature: float, top_p: float) -> BaseChatModel:
        """Instantiate the LLM for the given *provider* name.

        Args:
            provider:    Provider key, e.g. ``"azure"`` or ``"gemini"``.
            temperature: Passed through to the underlying model.
            top_p:       Passed through to the underlying model.

        Raises:
            ValueError: When *provider* has no registered implementation.
        """
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[provider]().build(temperature, top_p)
