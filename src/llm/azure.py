"""azure.py – Azure OpenAI LLM provider."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src.llm.base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """Builds an ``AzureChatOpenAI`` model from environment variables.

    Required env vars:
        AZURE_OPENAI_DEPLOYMENT  – model deployment name (e.g. "gpt-4o")
        AZURE_OPENAI_ENDPOINT    – e.g. https://<resource>.cognitiveservices.azure.com/
        AZURE_OPENAI_API_KEY     – API key
        AZURE_OPENAI_API_VERSION – optional, defaults to "2024-02-01"
    """

    def build(self, temperature: float, top_p: float) -> BaseChatModel:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
            top_p=top_p,
        )
