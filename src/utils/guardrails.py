"""guardrails.py - Request/response guardrail middleware for the LangGraph agent.

Implements a :class:`GuardrailsMiddleware` (Starlette ``BaseHTTPMiddleware``)
that sits in front of every ``POST /chat`` call and enforces:

Input guardrails (applied to ``ChatRequest.message`` before the agent runs)
  - Blocked keyword / phrase patterns (prompt injection, jailbreaks)
  - Basic PII detection (email, phone, SSN patterns)
  - Maximum message length

Output guardrails (applied to ``ChatResponse.reply`` before it is returned)
  - Blocked phrases in the model reply
  - PII redaction in the reply

All rule sets are configurable via environment variables or by subclassing.

Environment variables
---------------------
GUARDRAIL_MAX_INPUT_LEN    Maximum characters in the user message (default: 4096)
GUARDRAIL_BLOCK_PII_INPUT  "true" to reject messages that look like PII (default: true)
GUARDRAIL_REDACT_PII_OUTPUT "true" to redact PII patterns in agent replies (default: true)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable limits
# ---------------------------------------------------------------------------

_MAX_INPUT_LEN: int = int(os.getenv("GUARDRAIL_MAX_INPUT_LEN", "4096"))
_BLOCK_PII_INPUT: bool = os.getenv("GUARDRAIL_BLOCK_PII_INPUT", "true").lower() == "true"
_REDACT_PII_OUTPUT: bool = os.getenv("GUARDRAIL_REDACT_PII_OUTPUT", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# Prompt-injection / jailbreak triggers (case-insensitive, partial match)
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(?:\w+\s+){0,3}instructions?",   # "ignore [all] [previous] instructions"
        r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil|unrestricted)",
        r"disregard\s+(?:\w+\s+){0,3}(?:rules?|guidelines?|restrictions?|training)",
        r"pretend\s+(you\s+are|to\s+be)\s+(?:a\s+)?(?:human|person|unrestricted)",
        r"act\s+as\s+(?:if\s+you\s+(?:are|have\s+no))",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"system\s*prompt\s*[:=]",
        r"<\s*system\s*>",        # XML-style system injection
        r"\bDAN\b",               # "Do Anything Now" acronym
        r"tell\s+me\s+(your\s+)?system\s+prompt",
    ]
]

# PII patterns used for both input blocking and output redaction
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # US phone  (+1 optional, various separators)
    (re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"), "[PHONE]"),
    # US SSN
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),
    # Credit card (basic 13-19 digit runs)
    (re.compile(r"\b(?:\d[ \-]?){13,19}\b"), "[CARD]"),
]

# Phrases that should never appear in outgoing replies
_OUTPUT_BLOCKLIST: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(my\s+)?system\s+prompt\s+is",
        r"as\s+an?\s+unrestricted",
        r"i\s+(can|will)\s+ignore\s+(my\s+)?guidelines",
    ]
]

# ---------------------------------------------------------------------------
# Core check functions
# ---------------------------------------------------------------------------


def _check_injection(text: str) -> str | None:
    """Return the matched pattern string if injection is detected, else None."""
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def _contains_pii(text: str) -> bool:
    """Return True if the text contains a PII pattern."""
    return any(p.search(text) for p, _ in _PII_PATTERNS)


def _redact_pii(text: str) -> str:
    """Replace PII tokens in *text* with placeholder strings."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _check_output_blocklist(text: str) -> str | None:
    """Return the matched pattern string if the reply violates output rules."""
    for pattern in _OUTPUT_BLOCKLIST:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class GuardrailsMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware that enforces input and output guardrails
    on ``POST /chat`` requests.

    All other paths pass through unmodified.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Only guard the /chat endpoint
        if request.method == "POST" and request.url.path == "/chat":
            return await self._guard_chat(request, call_next)
        return await call_next(request)

    # -- input ---------------------------------------------------------------

    async def _guard_chat(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Read body and reconstruct the request so the downstream route can
        # also read it (BaseHTTPMiddleware consumes the receive() stream once).
        body_bytes = await request.body()

        async def _replay_receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request = Request(request.scope, _replay_receive)

        try:
            body = json.loads(body_bytes)
            message: str = body.get("message", "")
        except (json.JSONDecodeError, AttributeError):
            message = ""

        # 1. Length check
        if len(message) > _MAX_INPUT_LEN:
            logger.warning("Guardrail: input too long (%d chars)", len(message))
            return _blocked(
                "input_too_long",
                f"Message exceeds maximum allowed length of {_MAX_INPUT_LEN} characters.",
            )

        # 2. Prompt injection / jailbreak check
        matched = _check_injection(message)
        if matched:
            logger.warning("Guardrail: injection detected: %r", matched)
            return _blocked("prompt_injection", "Message contains disallowed content.")

        # 3. PII in input
        if _BLOCK_PII_INPUT and _contains_pii(message):
            logger.warning("Guardrail: PII detected in input")
            return _blocked(
                "pii_detected",
                "Message appears to contain personal information (PII). "
                "Please remove it before sending.",
            )

        # ---- Input passed — forward to the route ----
        response = await call_next(request)

        # -- output ----------------------------------------------------------
        # Only inspect JSON 200 responses from the /chat handler
        if response.status_code == 200 and "application/json" in response.headers.get(
            "content-type", ""
        ):
            raw = b""
            async for chunk in response.body_iterator:
                raw += chunk

            try:
                payload = json.loads(raw)
                reply: str = payload.get("reply", "")

                # 4. Output blocklist
                matched_out = _check_output_blocklist(reply)
                if matched_out:
                    logger.warning("Guardrail: blocked output phrase: %r", matched_out)
                    payload["reply"] = (
                        "I'm sorry, I can't provide that information."
                    )

                # 5. PII redaction in output
                elif _REDACT_PII_OUTPUT:
                    redacted = _redact_pii(reply)
                    if redacted != reply:
                        logger.info("Guardrail: PII redacted from output")
                    payload["reply"] = redacted

                raw = json.dumps(payload).encode()
            except (json.JSONDecodeError, AttributeError):
                pass  # non-JSON body; pass through as-is

            headers = dict(response.headers)
            headers["content-length"] = str(len(raw))
            return Response(
                content=raw,
                status_code=response.status_code,
                headers=headers,
                media_type="application/json",
            )

        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blocked(code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "guardrail_violation", "code": code, "detail": detail},
    )
