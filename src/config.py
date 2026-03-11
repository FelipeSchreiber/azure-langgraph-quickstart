"""config.py – Helpers to load and expose agent_config.yaml settings."""

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
    type: str = "none"            # "none" | "bearer" | "cert"
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
class MCPConfig:
    servers: List[MCPServer] = field(default_factory=list)


@dataclass
class Metadata:
    owner: str
    version: str
    created_at: str


@dataclass
class AgentConfig:
    agent_id: str
    agent_name: str
    deployment: str
    instructions: str
    agent_description: str
    temperature: float
    top_p: float
    mcp: MCPConfig
    metadata: Metadata


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    """Load *agent_config.yaml* and return a typed :class:`AgentConfig`.

    Args:
        path: Explicit path to the YAML file.  When *None* the function
              looks for ``agent_config.yaml`` next to this file's parent
              directory, then the current working directory.

    Returns:
        A fully populated :class:`AgentConfig` instance.

    Raises:
        FileNotFoundError: When the config file cannot be located.
        ValueError: When required fields are missing in the YAML.
    """
    if path is None:
        # Search heuristics: repo root (two levels up) → cwd
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

    # --- MCP servers ---
    mcp_raw = raw.get("mcp", {})
    servers: List[MCPServer] = []
    for srv in mcp_raw.get("servers", []):
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

    # --- Metadata ---
    meta_raw = raw.get("metadata", {})
    metadata = Metadata(
        owner=meta_raw.get("owner", ""),
        version=meta_raw.get("version", ""),
        created_at=meta_raw.get("created_at", ""),
    )

    return AgentConfig(
        agent_id=raw["agent_id"],
        agent_name=raw["agent_name"],
        deployment=raw["deployment"],
        instructions=raw["instructions"],
        agent_description=raw["agent_description"],
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        mcp=MCPConfig(servers=servers),
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
