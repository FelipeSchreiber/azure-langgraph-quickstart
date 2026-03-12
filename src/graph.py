"""graph.py – LangGraph agent built via the factory pattern.

Topology
--------
START  → llm_call  → (tool calls?) → call_tool → llm_call
                   ↘                             → END

``AgentFactory.create()`` is an async factory method that wires together the
MCP client, LLM (via ``LLMFactory``), node functions, and ``StateGraph``,
returning a compiled graph ready for invocation.

Which LLM provider is used is driven entirely by ``agent_config.yaml``::

    llm:
      provider: "azure"   # or "gemini", or any registered custom provider
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Dict, List

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from src.config import AgentConfig, get_config
from src.llm import LLMFactory
from src.utils.tracing import OTelCallbackHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Constructs a compiled LangGraph agent from an :class:`AgentConfig`.

    The factory is the single place responsible for:
    - resolving MCP server connectivity
    - delegating LLM instantiation to :class:`~src.llm.LLMFactory`
    - defining node functions and graph topology
    - compiling and returning the graph

    Usage::

        compiled = await AgentFactory.create()           # uses get_config()
        compiled = await AgentFactory.create(my_config)  # explicit config
        result   = await compiled.ainvoke({"messages": [...]})
    """

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _mcp_server_config(config: AgentConfig) -> Dict[str, Any]:
        """Convert AgentConfig MCP servers → MultiServerMCPClient dict."""
        servers: Dict[str, Any] = {}
        for srv in config.mcp.servers:
            cfg: Dict[str, Any] = {
                "url": srv.transport.mcp_url,
                "transport": srv.transport.type,
            }
            token = srv.resolve_token()
            if token:
                cfg["headers"] = {"Authorization": f"Bearer {token}"}
            servers[srv.name] = cfg
        return servers

    @staticmethod
    def _compile_graph(llm_with_tools: Any, tool_node: ToolNode, instructions: str):
        """Define nodes + edges and return a compiled StateGraph."""

        async def llm_call(s: AgentState) -> Dict[str, Any]:
            msgs: List[BaseMessage] = list(s["messages"])
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=instructions)] + msgs
            response: AIMessage = await llm_with_tools.ainvoke(msgs)
            return {"messages": [response]}

        builder = StateGraph(AgentState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", tools_condition)
        builder.add_edge("tools", "llm_call")
        graph = builder.compile()
        graph.callbacks = CallbackManager([OTelCallbackHandler()])
        return graph

    # -- public factory method -----------------------------------------------

    @classmethod
    async def create(cls, config: AgentConfig | None = None):
        """Async factory: fetch MCP tools, build LLM, compile the graph.

        Args:
            config: Pre-loaded config; defaults to the module-level singleton
                    returned by :func:`~src.config.get_config`.

        Returns:
            A compiled LangGraph ``StateGraph`` ready for ``ainvoke``.
        """
        if config is None:
            config = get_config()

        # 1. MCP tools
        mcp = MultiServerMCPClient(cls._mcp_server_config(config))
        tools = await mcp.get_tools()
        logger.info("MCP tools loaded (%d): %s", len(tools), [t.name for t in tools])

        # 2. LLM – provider resolved from config.llm.provider
        llm = LLMFactory.create(config.llm.provider, config.temperature, config.top_p)
        llm_with_tools = llm.bind_tools(tools)

        # 3. Graph
        return cls._compile_graph(llm_with_tools, ToolNode(tools), config.instructions)


# ---------------------------------------------------------------------------
# Public product – AgentGraph
# ---------------------------------------------------------------------------


class AgentGraph:
    """Lightweight public handle; delegates construction to AgentFactory."""

    async def ainvoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        compiled = await AgentFactory.create()
        return await compiled.ainvoke(state, config=config)

    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state, config=config))


# Module-level instance imported by app.py
graph = AgentGraph()
