"""graph.py LangGraph agent with MultiServerMCPClient.

Topology
--------
START  → llm_call  → (tool calls?) → call_tool → llm_call
                   ↘                             → END

Tools are fetched from all configured MCP servers via
``langchain_mcp_adapters.client.MultiServerMCPClient`` on every invocation.
``ToolNode`` from ``langgraph.prebuilt`` handles dispatch automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, Any, Dict, List, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import AgentConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# MCP config builder
# ---------------------------------------------------------------------------


def _build_mcp_config(config: AgentConfig) -> Dict[str, Any]:
    """Convert AgentConfig MCP servers to MultiServerMCPClient format."""
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


# ---------------------------------------------------------------------------
# Core async runner – opens MCP client, compiles & runs graph per invocation
# ---------------------------------------------------------------------------


async def _run_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    config = get_config()
    mcp_cfg = _build_mcp_config(config)

    mcp = MultiServerMCPClient(mcp_cfg)
    tools = await mcp.get_tools()
    logger.info(
        "MCP tools loaded (%d): %s", len(tools), [t.name for t in tools]
    )

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=config.temperature,
        top_p=config.top_p,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- nodes ---
    async def llm_call(s: AgentState) -> Dict[str, Any]:
        msgs: List[BaseMessage] = list(s["messages"])
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=config.instructions)] + msgs
        response: AIMessage = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    def should_call_tool(s: AgentState) -> Literal["call_tool", "__end__"]:
        last = s["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "call_tool"
        return "__end__"

    # --- graph ---
    builder = StateGraph(AgentState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("call_tool", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_call_tool,
        {"call_tool": "call_tool", "__end__": END},
    )
    builder.add_edge("call_tool", "llm_call")

    compiled = builder.compile()
    return await compiled.ainvoke(state)


# ---------------------------------------------------------------------------
# Public interface – sync (invoke) and async (ainvoke)
# ---------------------------------------------------------------------------


class AgentGraph:
    """Thin wrapper providing sync and async invocation of the MCP agent."""

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await _run_agent(state)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))


# Module-level instance imported by app.py
graph = AgentGraph()
