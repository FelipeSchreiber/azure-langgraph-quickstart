"""tools/rag.py - RAG retrieval tool builder.

Exposes a single ``retrieve_context(query: str)`` StructuredTool that calls the
RAG endpoint configured in ``agent_config.yaml``.

Supported backends
------------------
type: http
    POST ``rag.endpoint`` with ``{"query": ..., "top_k": ...}``
    and concatenates the returned ``documents[].content`` fields.

type: azure_search
    Calls Azure AI Search REST API with the query, using the configured
    index and API key.

If RAG is disabled (``rag.enabled: false``) the function returns ``None``
and the graph simply omits the tool.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class RetrieveInput(BaseModel):
    query: str = Field(..., description="Search query to retrieve relevant context for.")


# ---------------------------------------------------------------------------
# HTTP RAG backend
# ---------------------------------------------------------------------------


async def _call_http_rag(config: RAGConfig, query: str) -> str:
    """Call a generic HTTP RAG endpoint and return concatenated results."""
    headers = {"Content-Type": "application/json"}
    headers.update(config.authentication.resolve_headers())

    payload = {"query": query, "top_k": config.top_k}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(config.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("RAG HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return f"[RAG unavailable: HTTP {exc.response.status_code}]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG call failed: %s", exc)
        return f"[RAG unavailable: {exc}]"

    # Normalise response shape: support {"documents": [...]} and {"results": [...]}
    for key in ("documents", "results", "hits", "value"):
        items = data.get(key)
        if items and isinstance(items, list):
            chunks = []
            for item in items:
                if isinstance(item, dict):
                    text = item.get("content", item.get("text", item.get("chunk", "")))
                else:
                    text = str(item)
                if text:
                    chunks.append(text.strip())
            return "\n\n---\n\n".join(chunks) if chunks else "[No relevant context found]"

    # Fallback: return raw JSON
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Azure AI Search backend
# ---------------------------------------------------------------------------


async def _call_azure_search(config: RAGConfig, query: str) -> str:
    """Call Azure AI Search REST API and return concatenated result excerpts."""
    headers = config.authentication.resolve_headers()
    headers["Content-Type"] = "application/json"

    endpoint = config.endpoint.rstrip("/")
    index = config.index_name
    url = f"{endpoint}/indexes/{index}/docs/search?api-version=2023-11-01"

    payload = {
        "search": query,
        "top": config.top_k,
        "queryType": "semantic",
        "semanticConfiguration": "default",
        "captions": "extractive",
        "answers": "extractive",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Azure Search error %s: %s", exc.response.status_code, exc.response.text
        )
        return f"[RAG unavailable: HTTP {exc.response.status_code}]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Azure Search call failed: %s", exc)
        return f"[RAG unavailable: {exc}]"

    chunks = []
    for item in data.get("value", []):
        # Prefer the semantic caption highlight; fall back to full content field
        caption = (
            item.get("@search.captions", [{}])[0].get("highlights")
            or item.get("@search.captions", [{}])[0].get("text")
            or item.get("content", "")
        )
        if caption:
            chunks.append(caption.strip())
    return "\n\n---\n\n".join(chunks) if chunks else "[No relevant context found]"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_rag_tool(config: RAGConfig) -> Optional[StructuredTool]:
    """Build a ``retrieve_context`` StructuredTool from the RAG config.

    Args:
        config: :class:`~src.config.RAGConfig` parsed from ``agent_config.yaml``.

    Returns:
        A :class:`StructuredTool` when RAG is enabled, ``None`` otherwise.
    """
    if not config.enabled:
        return None

    async def _retrieve(query: str) -> str:
        if config.type == "azure_search":
            return await _call_azure_search(config, query)
        return await _call_http_rag(config, query)

    def _retrieve_sync(query: str) -> str:
        import asyncio
        return asyncio.run(_retrieve(query))

    return StructuredTool(
        name="retrieve_context",
        description=(
            "Retrieve relevant background context from the knowledge base. "
            "Use this tool BEFORE answering questions that require factual knowledge, "
            "recent information, or domain-specific data not in your training set. "
            "Input: a concise search query string."
        ),
        args_schema=RetrieveInput,
        func=_retrieve_sync,
        coroutine=_retrieve,
    )
