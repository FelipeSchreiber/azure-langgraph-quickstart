"""gemini.py - Google Gemini LLM provider."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Builds a ``ChatGoogleGenerativeAI`` model from environment variables.

    Required env vars:
        GOOGLE_API_KEY  - Google AI Studio or Vertex AI API key

    Optional env vars:
        GEMINI_MODEL    - model name, defaults to "gemini-1.5-pro"

    Install extra:
        pip install langchain-google-genai
    """

    def build(self, temperature: float, top_p: float) -> BaseChatModel:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for the Gemini provider. "
                "Install it with: pip install 'langchain-google-genai>=2.0'"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=temperature,
            top_p=top_p,
        )
