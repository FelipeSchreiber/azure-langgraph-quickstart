"""config.py - Typed loader for agent_config.yaml.

Full YAML schema
----------------
agent_type:           react | tool_calling
agent_id:             str
agent_name:           str
agent_description:    str
framework_type:       langgraph | langchain   (default: langgraph)

metadata:
  owner / version / created_at

llm:
  provider:           azure | gemini | ibm
  temperature:        float
  top_p:              float

prompt:
  system:             str
  input_variables:    list[str]

tools:
  mcp_servers:
    - name / description / transport / authentication
  openapi_servers:
    - name / description / spec_url / base_url / authentication

memory:
  type:               postgres | sqlite | memory
  connection_env_var: str           (postgres)
  db_path:            str           (sqlite)
  table_name:         str

rag:
  enabled:            bool
  type:               http | azure_search
  endpoint:           str
  top_k:              int
  index_name:         str           (azure_search)
  authentication:
    type:             none | bearer | api_key
    token_env_var:    str
    header_name:      str

guardrails:
  max_input_length:   int
  block_pii_input:    bool
  redact_pii_output:  bool
  custom_input_patterns:
    - pattern / description
  custom_output_patterns:
    - pattern / description

judge:
  enabled:            bool
  endpoint:           str           (optional remote scorer)
  use_agent_llm:      bool
  model:              str
  threshold:          float
  check_output:       bool
  max_retries:        int
  criteria:           str
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Transport / Auth primitives  (shared by MCP + OpenAPI servers)
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
    type: str = "none"                    # none | bearer | cert
    token_env_var: Optional[str] = None
    cert_path: Optional[str] = None
    key_path: Optional[str] = None

    def resolve_token(self) -> Optional[str]:
        if self.type != "bearer" or not self.token_env_var:
            return None
        return os.getenv(self.token_env_var)


@dataclass
class OpenAPIAuthentication:
    """Authentication config for OpenAPI server tools."""
    type: str = "none"                    # none | bearer | api_key | basic
    token_env_var: Optional[str] = None   # bearer / api_key secret
    header_name: Optional[str] = None     # api_key header name
    username_env_var: Optional[str] = None
    password_env_var: Optional[str] = None

    def resolve_token(self) -> Optional[str]:
        if self.type in ("bearer", "api_key") and self.token_env_var:
            return os.getenv(self.token_env_var)
        return None

    def resolve_basic(self) -> tuple[str, str] | None:
        if self.type == "basic" and self.username_env_var and self.password_env_var:
            u = os.getenv(self.username_env_var, "")
            p = os.getenv(self.password_env_var, "")
            return (u, p)
        return None


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


@dataclass
class MCPServer:
    name: str
    description: str
    transport: MCPTransport
    authentication: Optional[MCPAuthentication] = None

    def resolve_token(self) -> Optional[str]:
        if self.authentication is None:
            return None
        return self.authentication.resolve_token()


# ---------------------------------------------------------------------------
# OpenAPI server
# ---------------------------------------------------------------------------


@dataclass
class OpenAPIServer:
    """Describes a REST API exposed as tools via its OpenAPI spec."""
    name: str
    description: str
    spec_url: str                         # URL or local file path to the spec
    base_url: str
    authentication: Optional[OpenAPIAuthentication] = None


# ---------------------------------------------------------------------------
# Memory / Checkpointing
# ---------------------------------------------------------------------------


@dataclass
class MemoryConfig:
    """LangGraph checkpointer backend."""
    type: str = "none"                    # none (default) | memory | sqlite | postgres
    connection_env_var: Optional[str] = None   # postgres DSN env var
    db_path: Optional[str] = None         # sqlite file path
    table_name: str = "agent_checkpoints"

    def resolve_connection_string(self) -> Optional[str]:
        if self.type == "postgres" and self.connection_env_var:
            return os.getenv(self.connection_env_var)
        return None


# ---------------------------------------------------------------------------
# RAG endpoint
# ---------------------------------------------------------------------------


@dataclass
class RAGAuthentication:
    type: str = "none"                    # none | bearer | api_key
    token_env_var: Optional[str] = None
    header_name: Optional[str] = None    # api_key header

    def resolve_headers(self) -> dict:
        token = os.getenv(self.token_env_var or "", "")
        if self.type == "bearer" and token:
            return {"Authorization": f"Bearer {token}"}
        if self.type == "api_key" and token:
            hdr = self.header_name or "X-API-Key"
            return {hdr: token}
        return {}


@dataclass
class RAGConfig:
    enabled: bool = False
    type: str = "http"                    # http | azure_search
    endpoint: str = ""
    top_k: int = 5
    index_name: str = ""                  # azure_search only
    authentication: RAGAuthentication = field(default_factory=RAGAuthentication)


# ---------------------------------------------------------------------------
# Guardrails – custom regex patterns
# ---------------------------------------------------------------------------


@dataclass
class RegexPattern:
    pattern: str
    description: str = ""


@dataclass
class GuardrailsConfig:
    max_input_length: int = 4096
    block_pii_input: bool = True
    redact_pii_output: bool = True
    custom_input_patterns: List[RegexPattern] = field(default_factory=list)
    custom_output_patterns: List[RegexPattern] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    enabled: bool = False
    endpoint: Optional[str] = None       # remote HTTP scorer (optional)
    use_agent_llm: bool = True           # use the same LLM configured above
    model: str = "gpt-4o"               # relevant only when use_agent_llm=False
    threshold: float = 0.75
    check_output: bool = True
    max_retries: int = 2
    criteria: str = (
        "Rate the following AI response from 0.0 to 1.0 on: "
        "1. Factual accuracy and relevance. "
        "2. Absence of harmful content. "
        "3. Completeness and clarity. "
        'Respond with ONLY a JSON object: {"score": <float>, "reason": "<string>"}'
    )


# ---------------------------------------------------------------------------
# LLM / Prompt / Tools
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    provider: str = "azure"
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class PromptConfig:
    system: str = ""
    input_variables: List[str] = field(default_factory=lambda: ["messages"])


@dataclass
class ToolsConfig:
    mcp_servers: List[MCPServer] = field(default_factory=list)
    openapi_servers: List[OpenAPIServer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class Metadata:
    owner: str = ""
    version: str = ""
    created_at: str = ""


# ---------------------------------------------------------------------------
# Root AgentConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    # identity
    agent_id: str = ""
    agent_name: str = ""
    agent_description: str = ""
    agent_type: str = "react"
    framework_type: str = "langgraph"

    # sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    metadata: Metadata = field(default_factory=Metadata)

    # ------------------------------------------------------------------ #
    # Convenience aliases                                                  #
    # ------------------------------------------------------------------ #

    @property
    def instructions(self) -> str:
        return self.prompt.system

    @property
    def temperature(self) -> float:
        return self.llm.temperature

    @property
    def top_p(self) -> float:
        return self.llm.top_p

    @property
    def mcp(self) -> "_MCPCompat":
        """Backward-compat shim for code that accesses ``config.mcp.servers``."""
        return _MCPCompat(servers=self.tools.mcp_servers)


@dataclass
class _MCPCompat:
    """Legacy wrapper kept for backward compatibility."""
    servers: List[MCPServer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_auth_generic(raw: dict) -> MCPAuthentication:
    return MCPAuthentication(
        type=raw.get("type", "none"),
        token_env_var=raw.get("token_env_var"),
        cert_path=raw.get("cert_path"),
        key_path=raw.get("key_path"),
    )


def _parse_openapi_auth(raw: dict) -> OpenAPIAuthentication:
    return OpenAPIAuthentication(
        type=raw.get("type", "none"),
        token_env_var=raw.get("token_env_var"),
        header_name=raw.get("header_name"),
        username_env_var=raw.get("username_env_var"),
        password_env_var=raw.get("password_env_var"),
    )


def _parse_mcp_servers(raw_list: list) -> List[MCPServer]:
    servers: List[MCPServer] = []
    for srv in raw_list:
        t = srv.get("transport", {})
        a = srv.get("authentication", {})
        servers.append(
            MCPServer(
                name=srv["name"],
                description=srv.get("description", ""),
                transport=MCPTransport(
                    type=t.get("type", "http"),
                    endpoint=t["endpoint"],
                ),
                authentication=_parse_auth_generic(a) if a else None,
            )
        )
    return servers


def _parse_openapi_servers(raw_list: list) -> List[OpenAPIServer]:
    servers: List[OpenAPIServer] = []
    for srv in raw_list:
        a = srv.get("authentication", {})
        servers.append(
            OpenAPIServer(
                name=srv["name"],
                description=srv.get("description", ""),
                spec_url=srv["spec_url"],
                base_url=srv.get("base_url", ""),
                authentication=_parse_openapi_auth(a) if a else None,
            )
        )
    return servers


def _parse_memory(raw: dict) -> MemoryConfig:
    return MemoryConfig(
        type=raw.get("type", "none"),
        connection_env_var=raw.get("connection_env_var"),
        db_path=raw.get("db_path"),
        table_name=raw.get("table_name", "agent_checkpoints"),
    )


def _parse_rag(raw: dict) -> RAGConfig:
    a = raw.get("authentication", {})
    return RAGConfig(
        enabled=raw.get("enabled", False),
        type=raw.get("type", "http"),
        endpoint=raw.get("endpoint", ""),
        top_k=int(raw.get("top_k", 5)),
        index_name=raw.get("index_name", ""),
        authentication=RAGAuthentication(
            type=a.get("type", "none"),
            token_env_var=a.get("token_env_var"),
            header_name=a.get("header_name"),
        ),
    )


def _parse_guardrails(raw: dict) -> GuardrailsConfig:
    def _patterns(lst: list) -> List[RegexPattern]:
        return [
            RegexPattern(
                pattern=p["pattern"],
                description=p.get("description", ""),
            )
            for p in lst
        ]

    return GuardrailsConfig(
        max_input_length=int(raw.get("max_input_length", 4096)),
        block_pii_input=raw.get("block_pii_input", True),
        redact_pii_output=raw.get("redact_pii_output", True),
        custom_input_patterns=_patterns(raw.get("custom_input_patterns", [])),
        custom_output_patterns=_patterns(raw.get("custom_output_patterns", [])),
    )


def _parse_judge(raw: dict) -> JudgeConfig:
    return JudgeConfig(
        enabled=raw.get("enabled", False),
        endpoint=raw.get("endpoint"),
        use_agent_llm=raw.get("use_agent_llm", True),
        model=raw.get("model", "gpt-4o"),
        threshold=float(raw.get("threshold", 0.75)),
        check_output=raw.get("check_output", True),
        max_retries=int(raw.get("max_retries", 2)),
        criteria=raw.get("criteria", JudgeConfig.__dataclass_fields__["criteria"].default),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    """Load *agent_config.yaml* and return a typed :class:`AgentConfig`.

    Searches:
      1. Explicit *path* argument.
      2. ``agent_config.yaml`` at the project root (parent of ``src/``).
      3. Current working directory.
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

    # LLM
    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        provider=llm_raw.get("provider", "azure"),
        temperature=float(llm_raw.get("temperature", 0.7)),
        top_p=float(llm_raw.get("top_p", 0.95)),
    )

    # Prompt
    prompt_raw = raw.get("prompt", {})
    prompt = PromptConfig(
        system=prompt_raw.get("system", ""),
        input_variables=prompt_raw.get("input_variables", ["messages"]),
    )

    # Tools
    tools_raw = raw.get("tools", {})
    tools = ToolsConfig(
        mcp_servers=_parse_mcp_servers(tools_raw.get("mcp_servers", [])),
        openapi_servers=_parse_openapi_servers(tools_raw.get("openapi_servers", [])),
    )

    # Memory
    memory = _parse_memory(raw.get("memory", {}))

    # RAG
    rag = _parse_rag(raw.get("rag", {}))

    # Guardrails
    guardrails = _parse_guardrails(raw.get("guardrails", {}))

    # Judge
    judge = _parse_judge(raw.get("judge", {}))

    # Metadata
    meta_raw = raw.get("metadata", {})
    metadata = Metadata(
        owner=meta_raw.get("owner", ""),
        version=meta_raw.get("version", ""),
        created_at=str(meta_raw.get("created_at", "")),
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
        memory=memory,
        rag=rag,
        guardrails=guardrails,
        judge=judge,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Return the cached :class:`AgentConfig`, loading it on first call."""
    global _config
    if _config is None:
        _config = load_agent_config()
    return _config
