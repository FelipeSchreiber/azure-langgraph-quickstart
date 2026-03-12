"""tracing.py - OpenTelemetry backend for the LangGraph agent.

Wires an OTel ``TracerProvider`` into LangChain's callback system so that
every LLM call, tool invocation, and chain step is recorded as a span.

Usage
-----
Call :func:`setup_tracing` once at application startup (e.g. in ``app.py``).
LangGraph node functions automatically pick up the global callbacks when
``config["callbacks"]`` is set on each ``ainvoke`` call, or install it
globally via :func:`get_callback_handler`.

Environment variables
---------------------
OTEL_EXPORTER_OTLP_ENDPOINT   Collector endpoint (default: http://localhost:4317)
OTEL_SERVICE_NAME              Service name attached to every span
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

_TRACER_NAME = __name__
_tracer: Optional[trace.Tracer] = None


def setup_tracing(
    service_name: str = "langgraph-agent",
    endpoint: Optional[str] = None,
) -> trace.Tracer:
    """Configure the global OTel ``TracerProvider`` and return a tracer.

    Safe to call multiple times; subsequent calls are no-ops if a real
    provider is already installed.

    Args:
        service_name: Logical name for this service, attached to every span.
                      Overridden by the ``OTEL_SERVICE_NAME`` env var when set.
        endpoint:     OTLP gRPC collector URL.
                      Defaults to ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
                      ``http://localhost:4317``.

    Returns:
        A :class:`opentelemetry.trace.Tracer` bound to *service_name*.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    resolved_service = os.getenv("OTEL_SERVICE_NAME", service_name)
    resolved_endpoint = endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
    )
    # HTTP exporter expects the full traces path
    if not resolved_endpoint.endswith("/v1/traces"):
        resolved_endpoint = resolved_endpoint.rstrip("/") + "/v1/traces"

    resource = Resource.create({"service.name": resolved_service})
    exporter = OTLPSpanExporter(endpoint=resolved_endpoint)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(_TRACER_NAME)
    logger.info(
        "OTel tracing initialised (service=%s, endpoint=%s)",
        resolved_service,
        resolved_endpoint,
    )
    return _tracer


def get_callback_handler() -> "OTelCallbackHandler":
    """Return a fresh :class:`OTelCallbackHandler` using the active tracer.

    :func:`setup_tracing` must be called before this.
    """
    return OTelCallbackHandler()


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class OTelCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records spans via OpenTelemetry.

    One span is opened per chain, LLM call, and tool call.  Spans are
    automatically ended (with error status on exceptions).
    """

    def __init__(self) -> None:
        super().__init__()
        self._spans: Dict[UUID, Any] = {}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _tracer() -> trace.Tracer:
        t = trace.get_tracer(_TRACER_NAME)
        return t

    def _start(self, run_id: UUID, name: str, attributes: Dict[str, str]) -> None:
        span = self._tracer().start_span(name)
        for k, v in attributes.items():
            span.set_attribute(k, v)
        self._spans[run_id] = span

    def _end(self, run_id: UUID, attributes: Optional[Dict[str, str]] = None) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        span.end()

    def _error(self, run_id: UUID, error: BaseException) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))
        span.end()

    # -- chain ---------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "chain")
        self._start(run_id, name, {"chain.inputs": str(inputs)})

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._end(run_id, {"chain.outputs": str(outputs)})

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._error(run_id, error)

    # -- llm -----------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "llm")
        self._start(run_id, name, {"llm.prompts": str(prompts)})

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        token_usage = getattr(
            getattr(response, "llm_output", None), "get", lambda *_: {}
        )("token_usage", {})
        attrs = {k: str(v) for k, v in token_usage.items()} if token_usage else {}
        self._end(run_id, attrs or None)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._error(run_id, error)

    # -- tool ----------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "tool")
        self._start(run_id, name, {"tool.input": input_str})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, {"tool.output": str(output)})

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._error(run_id, error)
