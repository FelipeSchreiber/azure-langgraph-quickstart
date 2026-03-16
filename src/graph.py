"""graph.py - Foundry-style LangGraph ReAct agent.

Architecture
------------

    START
      │
      ▼
  llm_call ──── (tool calls present) ──► tools ──► llm_call
      │
      │ (no tool calls → final answer)
      ▼
  judge_output  (only when judge.enabled)
      │
      ├── score ≥ threshold  ──────────────────────────────► END
      │
      └── score < threshold + retries left ──► llm_call (retry)
                │
                └── max retries reached ────────────────────► END

Features loaded from ``agent_config.yaml``
------------------------------------------
tools.mcp_servers      MCP HTTP tools via MultiServerMCPClient
tools.openapi_servers  OpenAPI spec tools (dynamically parsed)
rag                    ``retrieve_context`` tool calling the RAG endpoint
memory                 LangGraph checkpointer (memory | sqlite | postgres)
                       – enables persistent per-session conversation history
judge                  LLM-as-a-judge quality gate on final answers

Usage
-----
    from src.graph import graph          # module-level AgentGraph instance
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        thread_id="session-123",         # enables memory
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import AgentConfig, get_config
from src.judge.llm_judge import JudgeResult, evaluate_response
from src.llm import LLMFactory
from src.memory.checkpoint import build_checkpointer
from src.tools.openapi import build_openapi_tools
from src.tools.rag import build_rag_tool
from src.utils.tracing import OTelCallbackHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # How many times the judge has rejected the response in this turn
    retry_count: int
    # Set to True when the judge accepts the response, False when it rejects
    judge_passed: bool
    # Original user question preserved for judge context across retries
    original_question: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class AgentFactory:
    """Async factory that compiles the LangGraph ReAct graph with all integrations.

    Call :meth:`create` once at startup; the compiled object is cached and
    reused for every request.
    """

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _mcp_server_config(config: AgentConfig) -> Dict[str, Any]:
        """Convert AgentConfig MCP servers → MultiServerMCPClient dict."""
        servers: Dict[str, Any] = {}
        for srv in config.tools.mcp_servers:
            cfg: Dict[str, Any] = {
                "url": srv.transport.mcp_url,
                "transport": srv.transport.type,
            }
            token = srv.resolve_token()
            if token:
                cfg["headers"] = {"Authorization": f"Bearer {token}"}
            servers[srv.name] = cfg
        return servers

    @classmethod
    async def _load_all_tools(cls, config: AgentConfig) -> list:
        """Load MCP tools + OpenAPI tools + RAG tool."""
        all_tools: list = []

        # 1. MCP tools
        mcp_servers = cls._mcp_server_config(config)
        if mcp_servers:
            try:
                mcp_client = MultiServerMCPClient(mcp_servers)
                mcp_tools = await mcp_client.get_tools()
                all_tools.extend(mcp_tools)
                logger.info("MCP tools loaded (%d): %s", len(mcp_tools), [t.name for t in mcp_tools])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load MCP tools: %s", exc)

        # 2. OpenAPI tools
        for srv in config.tools.openapi_servers:
            oa_tools = await build_openapi_tools(srv)
            all_tools.extend(oa_tools)

        # 3. RAG tool
        rag_tool = build_rag_tool(config.rag)
        if rag_tool is not None:
            all_tools.append(rag_tool)
            logger.info("RAG tool registered (endpoint=%s)", config.rag.endpoint)

        logger.info(
            "Total tools available to agent: %d → %s",
            len(all_tools),
            [t.name for t in all_tools],
        )
        return all_tools

    # -- graph compilation ---------------------------------------------------

    @classmethod
    async def create(cls, config: AgentConfig | None = None) -> tuple[Any, Any]:
        """Build and compile the LangGraph agent.

        Returns:
            Tuple of (compiled_graph, checkpointer).
            The checkpointer must be kept alive for the lifecycle of the app.
        """
        if config is None:
            config = get_config()

        # 1. Tools
        tools = await cls._load_all_tools(config)

        # 2. LLM
        llm = LLMFactory.create(config.llm.provider, config.temperature, config.top_p)
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # 3. Checkpointer
        checkpointer = await build_checkpointer(config.memory)

        # 4. OTel callback
        otel_cb = OTelCallbackHandler()

        # ─────────────────────────────────────────────────────────── nodes ──

        instructions = config.instructions

        async def llm_call(state: AgentState) -> Dict[str, Any]:
            """Run the LLM with the current message history."""
            msgs: List[BaseMessage] = list(state["messages"])
            # Prepend system prompt if not already present
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=instructions)] + msgs

            # On retry: inject judge feedback as a HumanMessage nudge
            retry = state.get("retry_count", 0)
            if retry > 0:
                last = msgs[-1]
                if isinstance(last, AIMessage):
                    nudge = HumanMessage(
                        content=(
                            "Your previous answer did not meet quality standards. "
                            "Please revise it to be more accurate, complete, and clear."
                        )
                    )
                    msgs = msgs + [nudge]

            response: AIMessage = await llm_with_tools.ainvoke(
                msgs,
                config={"callbacks": [otel_cb]},
            )
            return {"messages": [response]}

        async def judge_output(state: AgentState) -> Dict[str, Any]:
            """Quality-gate the last AIMessage; route to retry or END."""
            messages = state["messages"]
            last_ai = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)), None
            )
            if last_ai is None:
                return {"retry_count": state.get("retry_count", 0)}

            question = state.get("original_question", "")
            response_text = (
                last_ai.content
                if isinstance(last_ai.content, str)
                else str(last_ai.content)
            )

            result: JudgeResult = await evaluate_response(
                question=question,
                response=response_text,
                config=config.judge,
                llm=llm,
            )
            logger.info("Judge: %s", result)

            new_retries = state.get("retry_count", 0) + (0 if result.passed else 1)
            return {"retry_count": new_retries, "judge_passed": result.passed}

        def _route_after_llm(state: AgentState) -> Literal["tools", "judge_output", "__end__"]:
            """Conditional edge: tools → llm, judge (if enabled), or END."""
            messages = state["messages"]
            last = messages[-1] if messages else None

            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"

            if config.judge.enabled and config.judge.check_output:
                return "judge_output"

            return END

        def _route_after_judge(state: AgentState) -> Literal["llm_call", "__end__"]:
            """Conditional edge from judge: retry when failed and retries remain, else END."""
            # judge_output sets judge_passed = True when quality threshold is met
            if state.get("judge_passed", True):
                return END

            retry = state.get("retry_count", 0)
            if retry >= config.judge.max_retries:
                logger.info("Judge: max retries (%d) reached – accepting as-is", retry)
                return END

            logger.info("Judge: retrying (attempt %d / %d)", retry, config.judge.max_retries)
            return "llm_call"

        # ─────────────────────────────────────────────────────── assemble ──

        builder = StateGraph(AgentState)

        builder.add_node("llm_call", llm_call)
        builder.add_node("tools", ToolNode(tools) if tools else _noop_tool_node)

        builder.add_edge(START, "llm_call")

        # Build conditional edge mapping dynamically so unreachable nodes
        # are never referenced when the judge is disabled.
        if config.judge.enabled:
            builder.add_node("judge_output", judge_output)
            builder.add_conditional_edges(
                "llm_call",
                _route_after_llm,
                {"tools": "tools", "judge_output": "judge_output", END: END},
            )
            builder.add_conditional_edges(
                "judge_output",
                _route_after_judge,
                {"llm_call": "llm_call", END: END},
            )
        else:
            # Without judge: llm_call → tools or END
            def _route_no_judge(state: AgentState) -> Literal["tools", "__end__"]:
                messages = state["messages"]
                last = messages[-1] if messages else None
                if isinstance(last, AIMessage) and last.tool_calls:
                    return "tools"
                return END

            builder.add_conditional_edges(
                "llm_call",
                _route_no_judge,
                {"tools": "tools", END: END},
            )

        builder.add_edge("tools", "llm_call")

        # Compile with checkpointer for persistent memory
        compiled = builder.compile(checkpointer=checkpointer)
        logger.info(
            "Agent compiled (framework=langgraph, memory=%s, judge=%s, tools=%d)",
            config.memory.type,
            config.judge.enabled,
            len(tools),
        )
        return compiled, checkpointer


async def _noop_tool_node(state: AgentState) -> AgentState:
    """Placeholder when no tools are loaded."""
    return state


# ---------------------------------------------------------------------------
# Public AgentGraph
# ---------------------------------------------------------------------------


class AgentGraph:
    """Framework-agnostic public handle; owns the compiled graph lifecycle.

    The underlying graph (including checkpointer) is built once on the first
    ``ainvoke`` call and reused for all subsequent requests.
    """

    _compiled: Any = None
    _checkpointer: Any = None
    _lock: asyncio.Lock | None = None

    @property
    def _init_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _ensure_compiled(self) -> None:
        if self._compiled is not None:
            return
        async with self._init_lock:
            if self._compiled is None:
                self._compiled, self._checkpointer = await AgentFactory.create()

    async def ainvoke(
        self,
        state: Dict[str, Any],
        thread_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke the agent and return the final state.

        Args:
            state:      Initial state dict, must include ``messages`` key.
            thread_id:  Session / thread identifier for conversation memory.
                        When provided the checkpointer loads / stores the
                        message history automatically.
            run_config: Additional LangGraph run config overrides.
        """
        await self._ensure_compiled()

        # Extract original question for judge context
        messages = state.get("messages", [])
        original_q = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                original_q = m.content if isinstance(m.content, str) else str(m.content)
                break

        enriched_state = {
            "messages": messages,
            "retry_count": 0,
            "judge_passed": True,
            "original_question": original_q,
        }

        config: Dict[str, Any] = dict(run_config or {})
        if thread_id:
            config.setdefault("configurable", {})["thread_id"] = thread_id

        return await self._compiled.ainvoke(
            enriched_state,
            config=config if config else None,
        )

    def invoke(
        self,
        state: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`ainvoke`."""
        return asyncio.run(self.ainvoke(state, thread_id=thread_id))


# Module-level singleton imported by app.py
graph = AgentGraph()
