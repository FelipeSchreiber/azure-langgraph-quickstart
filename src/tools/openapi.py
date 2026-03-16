"""tools/openapi.py - OpenAPI→MCP tools via fastMCP's FastMCPOpenAPI.

For each OpenAPI server declared in ``agent_config.yaml`` this module:

1. Fetches the OpenAPI spec (JSON or YAML) from ``spec_url``.
2. Creates an in-process ``FastMCPOpenAPI`` server (fastmcp ≥ 2.0).
3. Discovers available MCP tools via a one-shot in-process client connection.
4. Wraps each MCP tool as a LangChain ``StructuredTool``.
5. Each tool invocation opens a fresh in-process MCP client connection so
   no persistent socket is required.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
import yaml
from fastmcp import Client as FastMCPClient
from fastmcp.client import FastMCPTransport
from fastmcp.server.openapi import FastMCPOpenAPI
from langchain_core.tools import StructuredTool
from mcp.types import TextContent

from src.config import OpenAPIAuthentication, OpenAPIServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spec fetcher
# ---------------------------------------------------------------------------


async def _fetch_spec(spec_url: str) -> dict:
    """Fetch and parse an OpenAPI spec from a URL or local file path (async)."""
    p = Path(spec_url)
    if p.exists():
        raw = p.read_text(encoding="utf-8")
    else:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as c:
            resp = await c.get(spec_url)
            resp.raise_for_status()
            raw = resp.text

    if spec_url.endswith((".yaml", ".yml")):
        return yaml.safe_load(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _build_auth_headers(auth: OpenAPIAuthentication | None) -> Dict[str, str]:
    """Resolve authentication config to HTTP headers."""
    if auth is None:
        return {}
    if auth.type == "bearer":
        token = os.environ.get(auth.token_env_var, "") if auth.token_env_var else ""
        return {"Authorization": f"Bearer {token}"} if token else {}
    if auth.type == "api_key":
        token = os.environ.get(auth.token_env_var, "") if auth.token_env_var else ""
        header = auth.header_name or "X-API-Key"
        return {header: token} if token else {}
    if auth.type == "basic":
        import base64
        user = os.environ.get(auth.username_env_var, "") if auth.username_env_var else ""
        pwd = os.environ.get(auth.password_env_var, "") if auth.password_env_var else ""
        if user and pwd:
            encoded = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
    return {}


# ---------------------------------------------------------------------------
# Per-call coroutine factory
# ---------------------------------------------------------------------------


def _make_tool_coroutine(fmcp_server: FastMCPOpenAPI, tool_name: str):  # type: ignore[return]
    """Return a coroutine that opens a fresh in-process MCP connection per call.

    Using a fresh ``FastMCPTransport`` connection for every invocation means no
    persistent socket is required; the in-process overhead is negligible.
    """

    async def _coroutine(**kwargs: Any) -> str:
        async with FastMCPClient(FastMCPTransport(fmcp_server)) as client:
            result = await client.call_tool(tool_name, kwargs)
        # Flatten MCP content blocks to a plain string for LangChain
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else ""

    return _coroutine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_openapi_tools(server: OpenAPIServer) -> List[StructuredTool]:  # noqa: D401
    """Build LangChain StructuredTools from an OpenAPI server config via fastMCP.

    Steps:
        1. Fetch the OpenAPI spec from ``server.spec_url``.
        2. Create an in-process ``FastMCPOpenAPI`` server backed by an
           ``httpx.AsyncClient`` with any configured auth headers.
        3. Discover available tools via a one-shot in-process MCP connection.
        4. Wrap each MCP tool as a ``StructuredTool`` (fresh connection per call).

    Args:
        server: OpenAPI server configuration from ``agent_config.yaml``.

    Returns:
        List of LangChain tools ready to bind to the LLM.
        Returns ``[]`` on spec-fetch failure (error is logged).
    """
    logger.info("Loading OpenAPI spec for '%s' from %s", server.name, server.spec_url)
    try:
        spec = await _fetch_spec(server.spec_url)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch OpenAPI spec for '%s': %s", server.name, exc)
        return []

    auth_headers = _build_auth_headers(server.authentication)
    http_client = httpx.AsyncClient(
        base_url=server.base_url or "",
        headers=auth_headers,
        follow_redirects=True,
        timeout=30.0,
    )

    fmcp_server = FastMCPOpenAPI(
        openapi_spec=spec,
        client=http_client,
        name=server.name,
    )

    # Discover the tool catalogue via a one-shot in-process connection
    async with FastMCPClient(FastMCPTransport(fmcp_server)) as client:
        mcp_tools = await client.list_tools()

    lang_tools: List[StructuredTool] = []
    for mcp_tool in mcp_tools:
        lang_tools.append(
            StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                args_schema=mcp_tool.inputSchema,
                coroutine=_make_tool_coroutine(fmcp_server, mcp_tool.name),
            )
        )
        logger.debug("Registered OpenAPI tool: %s", mcp_tool.name)

    logger.info(
        "OpenAPI server '%s' → %d tools: %s",
        server.name,
        len(lang_tools),
        [t.name for t in lang_tools],
    )
    return lang_tools

