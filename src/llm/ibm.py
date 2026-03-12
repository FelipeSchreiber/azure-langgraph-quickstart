"""ibm.py – IBM watsonx.ai LLM provider."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src.llm.base import LLMProvider


class IBMWatsonxProvider(LLMProvider):
    """Builds a ``ChatWatsonx`` model from environment variables.

    Required env vars:
        WATSONX_API_KEY     – IBM Cloud API key
        WATSONX_PROJECT_ID  – watsonx.ai project ID
        WATSONX_URL         – service endpoint, e.g.
                              https://us-south.ml.cloud.ibm.com

    Optional env vars:
        WATSONX_MODEL       – model ID, defaults to
                              "meta-llama/llama-3-3-70b-instruct"

    Install extra:
        pip install langchain-ibm
    """

    def build(self, temperature: float, top_p: float) -> BaseChatModel:
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError as exc:
            raise ImportError(
                "langchain-ibm is required for the IBM provider. "
                "Install it with: pip install 'langchain-ibm>=0.3'"
            ) from exc

        return ChatWatsonx(
            model_id=os.getenv("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct"),
            url=os.environ["WATSONX_URL"],
            apikey=os.environ["WATSONX_API_KEY"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            params={
                "temperature": temperature,
                "top_p": top_p,
            },
        )
