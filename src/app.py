"""app.py – FastAPI application exposing the LangGraph agent.

Routes
------
POST /chat          – Send a message and receive the agent's reply.
GET  /metrics       – Basic runtime metrics (uptime, request count).
GET  /health/live   – Liveness probe: returns 200 when the process is up.
GET  /health/ready  – Readiness probe: returns 200 when dependencies are ready.

Swagger UI is available at /docs (built into FastAPI by default).
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

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

config = get_config()

app = FastAPI(
    title=config.agent_name,
    description=config.agent_description,
    version=config.metadata.version or "0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_START_TIME = time.time()
_request_count: int = 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "What is 2 + 2?", "session_id": "demo-session"}]
        }
    }


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    tool_calls_made: int = 0


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    agent_id: str
    agent_name: str


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
    """Run the LangGraph agent with the provided *message* and return its reply.

    The agent may call one or more MCP tools before producing a final answer.
    """
    global _request_count
    _request_count += 1

    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
        }
        result = await graph.ainvoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = result.get("messages", [])
    reply = messages[-1].content if messages else ""

    # Count how many ToolMessages were produced (each represents one tool invocation)
    from langchain_core.messages import ToolMessage

    tool_calls_made = sum(1 for m in messages if isinstance(m, ToolMessage))

    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        tool_calls_made=tool_calls_made,
    )


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
    )


@app.get(
    "/health/live",
    summary="Liveness probe",
    tags=["Health"],
)
async def health_live() -> JSONResponse:
    """Liveness probe – returns 200 when the process is running."""
    return JSONResponse(content={"status": "alive"})


@app.get(
    "/health/ready",
    summary="Readiness probe",
    tags=["Health"],
)
async def health_ready() -> JSONResponse:
    """Readiness probe – checks that the config is loaded and the graph is compiled."""
    try:
        get_config()  # raises if YAML is missing or malformed
        _ = graph       # raises AttributeError if graph failed to compile
        return JSONResponse(content={"status": "ready"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "detail": str(exc)},
        )
