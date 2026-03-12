"""graph.py – Agent factory supporting LangGraph and LangChain frameworks.

The ``framework_type`` key in ``agent_config.yaml`` selects the backend:

    framework_type: langgraph   (default)
        Custom StateGraph topology:
        START → llm_call → (tools?) → call_tool → llm_call → END

    framework_type: langchain
        ``langchain.agents.create_agent`` – delegates graph construction
        to LangChain's built-in tool-calling agent loop.

Both paths return a compiled graph that accepts
``ainvoke({"messages": [...]})`` so ``AgentGraph`` is framework-agnostic.
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

# Module-level cache: built once per process, reused across requests.
_agent_cache: Dict[str, Any] = {}  # keyed by framework_type


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Compiles an agent graph for the framework declared in ``agent_config.yaml``.

    Dispatch table
    ~~~~~~~~~~~~~~
    ``framework_type: langgraph``  →  :meth:`_compile_langgraph`
        Hand-crafted ``StateGraph`` with explicit llm_call / tools nodes.

    ``framework_type: langchain``  →  :meth:`_compile_langchain`
        ``langchain.agents.create_agent`` tool-calling loop.

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
    def _compile_langgraph(llm_with_tools: Any, tool_node: ToolNode, instructions: str) -> Any:
        """Hand-crafted StateGraph: START → llm_call ⇄ tools → END."""

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
        compiled = builder.compile()
        compiled.callbacks = CallbackManager([OTelCallbackHandler()])
        logger.info("Agent compiled via langgraph StateGraph")
        return compiled

    @staticmethod
    def _compile_langchain(llm: Any, tools: list, instructions: str) -> Any:
        """Delegate to ``langchain.agents.create_agent`` tool-calling loop."""
        from langchain.agents import create_agent

        compiled = create_agent(
            model=llm,
            tools=tools,
            system_prompt=instructions,
        )
        logger.info("Agent compiled via langchain.agents.create_agent")
        return compiled

    # -- public factory method -----------------------------------------------

    @classmethod
    async def create(cls, config: AgentConfig | None = None) -> Any:
        """Async factory: build MCP tools + LLM, then compile for the configured framework.

        Args:
            config: Pre-loaded config; defaults to the module-level singleton
                    returned by :func:`~src.config.get_config`.

        Returns:
            A compiled graph ready for ``ainvoke({"messages": [...]})``.
        """
        if config is None:
            config = get_config()

        framework = config.framework_type  # "langgraph" | "langchain"

        # 1. MCP tools (shared by both frameworks)
        mcp = MultiServerMCPClient(cls._mcp_server_config(config))
        tools = await mcp.get_tools()
        logger.info("MCP tools loaded (%d): %s", len(tools), [t.name for t in tools])

        # 2. LLM
        llm = LLMFactory.create(config.llm.provider, config.temperature, config.top_p)

        # 3. Compile – dispatch on framework_type
        if framework == "langchain":
            return cls._compile_langchain(llm, tools, config.instructions)
        else:
            # langgraph (default)
            llm_with_tools = llm.bind_tools(tools)
            return cls._compile_langgraph(llm_with_tools, ToolNode(tools), config.instructions)


# ---------------------------------------------------------------------------
# Public product – AgentGraph
# ---------------------------------------------------------------------------


class AgentGraph:
    """Framework-agnostic public handle; caches the compiled graph.

    The underlying graph is built once on first ``ainvoke`` call and
    reused for all subsequent requests, regardless of framework.
    """

    _compiled: Any = None

    async def ainvoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._compiled is None:
            self._compiled = await AgentFactory.create()
        return await self._compiled.ainvoke(state, config=config)

    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state, config=config))


# Module-level instance imported by app.py
graph = AgentGraph()
