"""config.py – Helpers to load and expose agent_config.yaml settings.

YAML schema (current)
---------------------
agent_type:       react | tool_calling          (new)
agent_id:         str
agent_name:       str
agent_description: str
framework_type:   langgraph | langchain          (default: langgraph)
metadata:
  owner / version / created_at
llm:
  provider:       azure | gemini | ibm
  temperature:    float                          (moved from top-level)
  top_p:          float                          (moved from top-level)
prompt:
  system:         str                            (was top-level 'instructions')
  input_variables: list[str]
tools:
  source:         mcp
  names:          list[str]
  mcp_servers:                                   (was 'mcp.servers')
    - name / description / transport / authentication
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Data-classes that mirror the YAML schema
# ---------------------------------------------------------------------------


@dataclass
class MCPTransport:
    type: str
    endpoint: str

    @property
    def mcp_url(self) -> str:
        return self.endpoint


@dataclass
class MCPAuthentication:
    type: str = "none"                    # "none" | "bearer" | "cert"
    token_env_var: Optional[str] = None   # required when type=bearer
    cert_path: Optional[str] = None       # required when type=cert
    key_path: Optional[str] = None        # required when type=cert


@dataclass
class MCPServer:
    name: str
    description: str
    transport: MCPTransport
    authentication: Optional[MCPAuthentication] = None

    def resolve_token(self) -> Optional[str]:
        """Return the bearer token, or None when authentication is disabled."""
        if self.authentication is None or self.authentication.type != "bearer":
            return None
        if not self.authentication.token_env_var:
            return None
        return os.getenv(self.authentication.token_env_var)


@dataclass
class LLMConfig:
    """LLM provider and sampling parameters."""
    provider: str = "azure"   # azure | gemini | ibm
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class PromptConfig:
    """System prompt and expected input variable names."""
    system: str = ""
    input_variables: List[str] = field(default_factory=lambda: ["messages"])


@dataclass
class ToolsConfig:
    """Tool source and MCP server list."""
    source: str = "mcp"
    names: List[str] = field(default_factory=list)
    mcp_servers: List[MCPServer] = field(default_factory=list)


@dataclass
class Metadata:
    owner: str = ""
    version: str = ""
    created_at: str = ""


@dataclass
class AgentConfig:
    # ---- identity ----------------------------------------------------------
    agent_id: str = ""
    agent_name: str = ""
    agent_description: str = ""
    agent_type: str = "react"             # react | tool_calling
    framework_type: str = "langgraph"     # langgraph | langchain

    # ---- sub-configs -------------------------------------------------------
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    metadata: Metadata = field(default_factory=Metadata)

    # ------------------------------------------------------------------ #
    # Backward-compat properties                                          #
    # Code that still reads config.instructions / .temperature / .top_p  #
    # / .mcp continues to work without modification.                      #
    # ------------------------------------------------------------------ #

    @property
    def instructions(self) -> str:
        """Alias for ``prompt.system``."""
        return self.prompt.system

    @property
    def temperature(self) -> float:
        """Alias for ``llm.temperature``."""
        return self.llm.temperature

    @property
    def top_p(self) -> float:
        """Alias for ``llm.top_p``."""
        return self.llm.top_p

    @property
    def mcp(self) -> "MCPConfig":
        """Alias for ``tools`` presented as an MCPConfig for legacy callers."""
        return MCPConfig(servers=self.tools.mcp_servers)


@dataclass
class MCPConfig:
    """Legacy wrapper kept for backward compatibility with graph.py callers."""
    servers: List[MCPServer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_mcp_servers(server_list: list) -> List[MCPServer]:
    servers: List[MCPServer] = []
    for srv in server_list:
        transport_raw = srv.get("transport", {})
        auth_raw = srv.get("authentication", {})
        servers.append(
            MCPServer(
                name=srv["name"],
                description=srv.get("description", ""),
                transport=MCPTransport(
                    type=transport_raw.get("type", "http"),
                    endpoint=transport_raw["endpoint"],
                ),
                authentication=MCPAuthentication(
                    type=auth_raw.get("type", "none"),
                    token_env_var=auth_raw.get("token_env_var"),
                    cert_path=auth_raw.get("cert_path"),
                    key_path=auth_raw.get("key_path"),
                ) if auth_raw else None,
            )
        )
    return servers


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    """Load *agent_config.yaml* and return a typed :class:`AgentConfig`.

    Args:
        path: Explicit path to the YAML file.  When *None* the function
              searches ``agent_config.yaml`` next to this file's parent
              directory, then the current working directory.

    Raises:
        FileNotFoundError: When the config file cannot be located.
    """
    if path is None:
        candidates = [
            Path(__file__).parent.parent / "agent_config.yaml",
            Path.cwd() / "agent_config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "agent_config.yaml not found. "
                f"Searched: {[str(c) for c in candidates]}"
            )

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    # --- LLM ---------------------------------------------------------------
    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        provider=llm_raw.get("provider", "azure"),
        temperature=float(llm_raw.get("temperature", 0.7)),
        top_p=float(llm_raw.get("top_p", 0.95)),
    )

    # --- Prompt ------------------------------------------------------------
    prompt_raw = raw.get("prompt", {})
    prompt = PromptConfig(
        system=prompt_raw.get("system", ""),
        input_variables=prompt_raw.get("input_variables", ["messages"]),
    )

    # --- Tools / MCP servers -----------------------------------------------
    tools_raw = raw.get("tools", {})
    mcp_servers = _parse_mcp_servers(tools_raw.get("mcp_servers", []))
    tools = ToolsConfig(
        source=tools_raw.get("source", "mcp"),
        names=tools_raw.get("names", []),
        mcp_servers=mcp_servers,
    )

    # --- Metadata ----------------------------------------------------------
    meta_raw = raw.get("metadata", {})
    metadata = Metadata(
        owner=meta_raw.get("owner", ""),
        version=meta_raw.get("version", ""),
        created_at=meta_raw.get("created_at", ""),
    )

    return AgentConfig(
        agent_id=raw.get("agent_id", ""),
        agent_name=raw.get("agent_name", ""),
        agent_description=raw.get("agent_description", ""),
        agent_type=raw.get("agent_type", "react"),
        framework_type=raw.get("framework_type", "langgraph"),
        llm=llm,
        prompt=prompt,
        tools=tools,
        metadata=metadata,
    )


# Singleton – loaded once at import time so all modules share the same object.
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Return the cached :class:`AgentConfig`, loading it on first call."""
    global _config
    if _config is None:
        _config = load_agent_config()
    return _config
