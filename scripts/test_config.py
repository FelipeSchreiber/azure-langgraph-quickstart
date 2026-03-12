#!/usr/bin/env python3
"""test_config.py – Smoke-test the config loader against a local MCP server.

Usage
-----
# Token is optional for servers that don't enforce auth locally
LOCAL_MCP_TOKEN=test-token python3 scripts/test_config.py

# Or without a token (the script will still probe the server)
python3 scripts/test_config.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

from src.config import load_agent_config, MCPServer

SEPARATOR = "-" * 60


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


# ---------------------------------------------------------------------------
# MCP JSON-RPC helper (mirrors graph.py)
# ---------------------------------------------------------------------------

def _mcp_jsonrpc(url: str, method: str, params: dict | None = None, headers: dict | None = None):
    """Send a single MCP JSON-RPC 2.0 request. Returns (result, session_id)."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    hdrs = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **(headers or {}),
    }
    resp = httpx.post(url, json=payload, headers=hdrs, timeout=15)
    resp.raise_for_status()
    session_id: str | None = resp.headers.get("mcp-session-id")

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    envelope = json.loads(data_str)
                    return envelope.get("result", envelope), session_id
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"No parseable data in SSE response: {resp.text[:200]}")
    else:
        envelope = resp.json()
        if "error" in envelope:
            raise RuntimeError(f"MCP error: {envelope['error']}")
        return envelope.get("result", envelope), session_id


def _mcp_init_session(url: str, base_headers: dict) -> str | None:
    """Initialize an MCP session and return the session ID."""
    _, session_id = _mcp_jsonrpc(
        url,
        "initialize",
        params={
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-config", "version": "0.1.0"},
        },
        headers=base_headers,
    )
    return session_id


def _auth_headers(server: MCPServer, token: str | None) -> dict:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------
def test_config_loads():
    print(SEPARATOR)
    print("1. Loading agent_config.yaml")
    cfg = load_agent_config()
    _ok(f"agent_id        = {cfg.agent_id}")
    _ok(f"agent_name      = {cfg.agent_name}")
    _ok(f"agent_type      = {cfg.agent_type}")
    _ok(f"framework_type  = {cfg.framework_type}")
    _ok(f"llm.provider    = {cfg.llm.provider}")
    _ok(f"llm.temperature = {cfg.llm.temperature}")
    _ok(f"llm.top_p       = {cfg.llm.top_p}")
    _ok(f"prompt.system   = {cfg.prompt.system[:60]!r}...")
    _ok(f"tools.names     = {cfg.tools.names}")
    _ok(f"mcp_servers     = {[s.name for s in cfg.tools.mcp_servers]}")
    _ok(f"metadata.version= {cfg.metadata.version}")
    return cfg


# ---------------------------------------------------------------------------
# 2. Authentication check
# ---------------------------------------------------------------------------
def test_token_resolution(server: MCPServer) -> str | None:
    print(SEPARATOR)
    auth = server.authentication
    auth_type = auth.type if auth else "none"
    print(f"2. Authentication type for '{server.name}': {auth_type}")
    token = server.resolve_token()
    if token:
        _ok(f"Bearer token resolved from ${auth.token_env_var}")
    else:
        _ok("No authentication required – sending requests without credentials")
    return token


# ---------------------------------------------------------------------------
# 3. HTTP connectivity (health check)
# ---------------------------------------------------------------------------
def test_server_reachable(server: MCPServer, token: str | None) -> bool:
    print(SEPARATOR)
    from urllib.parse import urlparse
    parsed = urlparse(server.transport.endpoint)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    print(f"3. Probing {base_url}/health")

    try:
        resp = httpx.get(f"{base_url}/health", headers=_auth_headers(server, token), timeout=5)
        _ok(f"GET /health  →  HTTP {resp.status_code}  body={resp.text[:80]!r}")
        return True
    except httpx.ConnectError:
        _fail("Connection refused – is the server running?")
        return False
    except Exception as exc:
        _fail(f"Unexpected error: {exc}")
        return False


# ---------------------------------------------------------------------------
# 4. MCP tools/list via JSON-RPC
# ---------------------------------------------------------------------------
def test_list_tools(server: MCPServer, token: str | None) -> list:
    print(SEPARATOR)
    mcp_url = server.transport.mcp_url
    print(f"4. tools/list  →  POST {mcp_url}")

    try:
        base_hdrs = _auth_headers(server, token)
        session_id = _mcp_init_session(mcp_url, base_hdrs)
        hdrs = {**base_hdrs, **({"Mcp-Session-Id": session_id} if session_id else {})}
        if session_id:
            _info(f"Session ID: {session_id}")
        result, _ = _mcp_jsonrpc(mcp_url, "tools/list", headers=hdrs)
        tools = result.get("tools", result) if isinstance(result, dict) else result
        if isinstance(tools, list) and tools:
            _ok(f"Found {len(tools)} tool(s):")
            for t in tools:
                name = t.get("name", "<unnamed>")
                desc = t.get("description", "")
                _info(f"  • {name}: {desc}")
        else:
            _ok("Server responded – no tools registered.")
            tools = []
        return tools
    except httpx.HTTPStatusError as exc:
        _fail(f"HTTP {exc.response.status_code}: {exc.response.text[:120]}")
        return []
    except Exception as exc:
        _fail(f"{exc}")
        return []


# ---------------------------------------------------------------------------
# 5. MCP tools/call via JSON-RPC
# ---------------------------------------------------------------------------
def test_call_tool(server: MCPServer, token: str | None, tools: list) -> None:
    print(SEPARATOR)
    print("5. tools/call  (dry-run with first available tool)")

    if not tools:
        _info("No tools to invoke – skipping.")
        return

    first_tool = tools[0]
    tool_name = first_tool.get("name", "")
    mcp_url = server.transport.mcp_url

    # Build minimal arguments from the input schema
    props = first_tool.get("inputSchema", {}).get("properties", {})
    sample_args: dict = {}
    for key, schema in props.items():
        t = schema.get("type", "string")
        sample_args[key] = 0 if t in ("number", "integer") else (False if t == "boolean" else "")

    _info(f"POST {mcp_url}  method=tools/call  name={tool_name}  args={sample_args}")

    try:
        base_hdrs = _auth_headers(server, token)
        session_id = _mcp_init_session(mcp_url, base_hdrs)
        hdrs = {**base_hdrs, **({"Mcp-Session-Id": session_id} if session_id else {})}
        result, _ = _mcp_jsonrpc(
            mcp_url,
            "tools/call",
            params={"name": tool_name, "arguments": sample_args},
            headers=hdrs,
        )
        # Normalise
        if isinstance(result, dict):
            content = result.get("content", result)
            if isinstance(content, list):
                text = " ".join(c.get("text", str(c)) for c in content)
            else:
                text = str(content)
        else:
            text = str(result)
        _ok(f"Result: {text[:200]}")
    except httpx.HTTPStatusError as exc:
        _fail(f"HTTP {exc.response.status_code}: {exc.response.text[:120]}")
    except Exception as exc:
        _fail(f"{exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nMCP Config Smoke Test")
    print(SEPARATOR)

    cfg = test_config_loads()

    if not cfg.mcp.servers:
        print("\nNo MCP servers defined – nothing to test.")
        sys.exit(0)

    exit_code = 0
    for server in cfg.mcp.servers:
        print(f"\n{'=' * 60}")
        print(f"Server : {server.name}")
        print(f"Base   : {server.transport.endpoint}")
        print(f"MCP    : {server.transport.mcp_url}")
        print(f"{'=' * 60}")

        token = test_token_resolution(server)
        reachable = test_server_reachable(server, token)

        if reachable:
            tools = test_list_tools(server, token)
            test_call_tool(server, token, tools)
        else:
            exit_code = 1

    print(SEPARATOR)
    print("Done.\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
