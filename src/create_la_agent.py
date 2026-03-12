#!/usr/bin/env python
"""create_la_agent.py – Build a LangGraph react agent and save its spec to la_agent.yaml.

Reads agent_config.yaml, instantiates a ``create_react_agent`` graph using
the configured LLM and MCP tools, then serialises the agent specification to
la_agent.yaml at the repo root.

Usage::

    python3 src/create_la_agent.py
    python3 src/create_la_agent.py --output path/to/la_agent.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure the repo root is on sys.path so ``src.*`` imports resolve correctly
# when this file is run directly as ``python3 src/create_la_agent.py``.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from langchain.agents import create_agent                  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

from src.config import get_config                          # noqa: E402
from src.llm.factory import LLMFactory                    # noqa: E402


async def _load_tools(config):
    """Connect to MCP servers and return the list of available tools."""
    servers: dict = {}
    for srv in config.mcp.servers:
        entry = {"url": srv.transport.mcp_url, "transport": srv.transport.type}
        token = srv.resolve_token()
        if token:
            entry["headers"] = {"Authorization": f"Bearer {token}"}
        servers[srv.name] = entry

    mcp = MultiServerMCPClient(servers)
    tools = await mcp.get_tools()
    print(f"  MCP tools loaded ({len(tools)}): {[t.name for t in tools]}")
    return tools


async def _build_agent(config, tools: list):
    """Instantiate the react agent (requires LLM env vars to be present)."""
    llm = LLMFactory.create(config.llm.provider, config.temperature, config.top_p)
    agent = create_agent(model=llm, tools=tools, system_prompt=config.instructions.strip())
    print("  Agent compiled successfully.")
    agent.save("la_agent.yaml")  # sanity check: can we save the spec after building?
    return agent


def _build_spec(config, tools: list) -> dict:
    """Return a serialisable dict describing the agent for la_agent.yaml."""
    mcp_servers = []
    for srv in config.mcp.servers:
        entry = {
            "name": srv.name,
            "description": srv.description,
            "transport": {"type": srv.transport.type, "endpoint": srv.transport.endpoint},
            "authentication": {"type": (srv.authentication.type if srv.authentication else "none")},
        }
        if srv.authentication and srv.authentication.token_env_var:
            entry["authentication"]["token_env_var"] = srv.authentication.token_env_var
        mcp_servers.append(entry)

    return {
        "agent_type": "react",
        "chain_type": "create_react_agent",
        "agent_id": config.agent_id,
        "agent_name": config.agent_name,
        "agent_description": config.agent_description,
        "metadata": {
            "owner": config.metadata.owner,
            "version": config.metadata.version,
            "created_at": config.metadata.created_at,
            "source_config": "agent_config.yaml",
        },
        "llm": {
            "provider": config.llm.provider,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        "prompt": {
            "system": config.instructions.strip(),
            "input_variables": ["messages"],
        },
        "tools": {
            "source": "mcp",
            "names": [t.name for t in tools],
            "mcp_servers": mcp_servers,
        },
    }


async def main_async(output: str) -> None:
    config = get_config()

    # Step 1: load MCP tools (works without LLM env vars)
    tools = await _load_tools(config)

    # Step 2: write the YAML spec
    spec = _build_spec(config, tools)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.dump(spec, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✓ la_agent.yaml written → {out_path.resolve()}")
    print(f"  agent_type : {spec['agent_type']} (create_react_agent)")
    print(f"  llm        : {spec['llm']['provider']}  "
          f"(temp={spec['llm']['temperature']}, top_p={spec['llm']['top_p']})")
    print(f"  tools      : {spec['tools']['names']}")

    # Step 3: try compiling the full agent (requires LLM env vars)
    try:
        await _build_agent(config, tools)
    except KeyError as exc:
        print(f"  ⚠  Agent compilation skipped — missing env var: {exc}")
        print("     (Run inside the Docker container to compile with a live LLM.)")


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "la_agent.yaml"),
        help="Destination file (default: la_agent.yaml at repo root)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.output))


if __name__ == "__main__":
    main()