"""app.py - FastAPI application exposing the LangGraph Foundry agent.

Routes
------
POST /chat          - Send a message; session_id activates conversation memory.
GET  /history/{id}  - Retrieve message history for a session (requires memory).
DELETE /history/{id}- Clear conversation history for a session.
GET  /metrics       - Basic runtime metrics (uptime, request count).
GET  /health/live   - Liveness probe.
GET  /health/ready  - Readiness probe.

Swagger UI is available at /docs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.config import get_config
from src.graph import graph
from src.utils.guardrails import GuardrailsMiddleware
from src.utils.tracing import setup_tracing

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

config = get_config()

app = FastAPI(
    title=config.agent_name,
    description=config.agent_description,
    version=config.metadata.version or "0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_START_TIME = time.time()
_request_count: int = 0

app.add_middleware(GuardrailsMiddleware)
setup_tracing(service_name=config.agent_name)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is the capital of France?",
                    "session_id": "user-42-session-1",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    tool_calls_made: int = 0
    judge_retries: int = 0


class MessageRecord(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageRecord]


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    agent_id: str
    agent_name: str
    memory_type: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the agent",
    tags=["Agent"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the LangGraph agent and return its reply.

    When a ``session_id`` is provided the agent uses its configured
    checkpointer to load prior conversation turns automatically, enabling
    multi-turn conversations that survive service restarts.
    """
    global _request_count
    _request_count += 1

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
    }

    try:
        result = await graph.ainvoke(initial_state, thread_id=request.session_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = result.get("messages", [])
    raw_content = messages[-1].content if messages else ""

    # Flatten list-type content blocks (vision models, etc.)
    if isinstance(raw_content, list):
        reply = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        ).strip()
    else:
        reply = str(raw_content)

    from langchain_core.messages import ToolMessage

    tool_calls_made = sum(1 for m in messages if isinstance(m, ToolMessage))
    judge_retries = result.get("retry_count", 0)

    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        tool_calls_made=tool_calls_made,
        judge_retries=judge_retries,
    )


@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Get conversation history for a session",
    tags=["Agent"],
)
async def get_history(session_id: str) -> HistoryResponse:
    """Return all messages for *session_id* from the checkpointer.

    Requires a persistent memory backend (sqlite or postgres).
    Returns an empty list when the session is not found or memory is
    in-process only.
    """
    await graph._ensure_compiled()

    checkpointer = graph._checkpointer
    if checkpointer is None:
        return HistoryResponse(session_id=session_id, messages=[])

    try:
        thread_config = {"configurable": {"thread_id": session_id}}
        # LangGraph checkpointers expose aget() returning the latest checkpoint
        state_snapshot = await checkpointer.aget(thread_config)
    except Exception:  # noqa: BLE001
        return HistoryResponse(session_id=session_id, messages=[])

    if state_snapshot is None:
        return HistoryResponse(session_id=session_id, messages=[])

    # state_snapshot is a dict (channel_values) at this checkpoint
    channel_values = getattr(state_snapshot, "channel_values", state_snapshot) or {}
    raw_messages = channel_values.get("messages", [])
    records: List[MessageRecord] = []
    for m in raw_messages:
        role = type(m).__name__.replace("Message", "").lower()
        content = m.content if isinstance(m.content, str) else str(m.content)
        records.append(MessageRecord(role=role, content=content))

    return HistoryResponse(session_id=session_id, messages=records)


@app.delete(
    "/history/{session_id}",
    summary="Clear conversation history for a session",
    tags=["Agent"],
)
async def delete_history(session_id: str) -> JSONResponse:
    """Delete all checkpointed state for *session_id*.

    Subsequent messages in that session start fresh.
    Requires a persistent memory backend; a 501 is returned for in-memory.
    """
    await graph._ensure_compiled()

    checkpointer = graph._checkpointer
    if checkpointer is None:
        return JSONResponse(
            status_code=501,
            content={"error": "no persistent memory backend is configured"},
        )

    # Try the official async delete API (available in langgraph >= 0.3)
    try:
        thread_config = {"configurable": {"thread_id": session_id}}
        delete_fn = getattr(checkpointer, "adelete_thread", None)
        if delete_fn is None:
            # Older API: adelete takes the thread config
            delete_fn = getattr(checkpointer, "adelete", None)
        if delete_fn is not None:
            await delete_fn(thread_config)
            return JSONResponse(content={"status": "cleared", "session_id": session_id})

        # Fallback: inform user that delete is not supported
        return JSONResponse(
            status_code=501,
            content={
                "error": "delete not supported by this checkpointer implementation",
                "hint": "Upgrade langgraph-checkpoint-sqlite / langgraph-checkpoint-postgres",
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime metrics",
    tags=["Operations"],
)
async def metrics() -> MetricsResponse:
    """Return basic runtime metrics for monitoring dashboards."""
    cfg = get_config()
    return MetricsResponse(
        uptime_seconds=round(time.time() - _START_TIME, 2),
        total_requests=_request_count,
        agent_id=cfg.agent_id,
        agent_name=cfg.agent_name,
        memory_type=cfg.memory.type,
    )


@app.get(
    "/health/live",
    summary="Liveness probe",
    tags=["Health"],
)
async def health_live() -> JSONResponse:
    """Returns 200 when the process is running."""
    return JSONResponse(content={"status": "alive"})


@app.get(
    "/health/ready",
    summary="Readiness probe",
    tags=["Health"],
)
async def health_ready() -> JSONResponse:
    """Checks that config is loaded and the graph has been initialised."""
    try:
        get_config()
        _ = graph
        return JSONResponse(content={"status": "ready"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "detail": str(exc)},
        )
