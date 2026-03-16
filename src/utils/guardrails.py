"""guardrails.py - Request/response guardrail middleware for the LangGraph agent.

Enforces input and output safety rules on every ``POST /chat`` call.

Rule sources (merged at startup, highest priority first)
---------------------------------------------------------
1. ``agent_config.yaml`` → ``guardrails.custom_input_patterns`` /
   ``guardrails.custom_output_patterns``  (operator-defined regex patterns)
2. Built-in injection / jailbreak patterns (always active)
3. Built-in PII patterns (blockable via config)

Settings (``agent_config.yaml`` keys take precedence over env vars)
--------------------------------------------------------------------
guardrails.max_input_length     Maximum message characters   (env: GUARDRAIL_MAX_INPUT_LEN)
guardrails.block_pii_input      Reject PII in input          (env: GUARDRAIL_BLOCK_PII_INPUT)
guardrails.redact_pii_output    Redact PII in replies        (env: GUARDRAIL_REDACT_PII_OUTPUT)
guardrails.custom_input_patterns   Additional input block patterns
guardrails.custom_output_patterns  Additional output block patterns
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Awaitable, Callable, List, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in pattern libraries
# ---------------------------------------------------------------------------

# Prompt-injection / jailbreak triggers (case-insensitive)
_BUILTIN_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(?:\w+\s+){0,3}instructions?",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil|unrestricted)",
    r"disregard\s+(?:\w+\s+){0,3}(?:rules?|guidelines?|restrictions?|training)",
    r"pretend\s+(you\s+are|to\s+be)\s+(?:a\s+)?(?:human|person|unrestricted)",
    r"act\s+as\s+(?:if\s+you\s+(?:are|have\s+no))",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"system\s*prompt\s*[:=]",
    r"<\s*system\s*>",
    r"\bDAN\b",
    r"tell\s+me\s+(your\s+)?system\s+prompt",
]

# PII patterns: (compiled-pattern, redaction-placeholder)
_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ \-]?){13,19}\b"), "[CARD]"),
]

# Built-in output blocklist
_BUILTIN_OUTPUT_PATTERNS: List[str] = [
    r"(my\s+)?system\s+prompt\s+is",
    r"as\s+an?\s+unrestricted",
    r"i\s+(can|will)\s+ignore\s+(my\s+)?guidelines",
]


# ---------------------------------------------------------------------------
# Runtime pattern registry  – merged from config + built-ins at first request
# ---------------------------------------------------------------------------

_compiled_injection: List[re.Pattern] | None = None
_compiled_output: List[re.Pattern] | None = None
_max_input_len: int | None = None
_block_pii_input: bool | None = None
_redact_pii_output: bool | None = None


def _load_config_patterns() -> None:
    """Lazy-initialise all patterns from agent_config.yaml + env vars.

    Called once the first time GuardrailsMiddleware processes a request,
    ensuring the config singleton is already populated.
    """
    global _compiled_injection, _compiled_output
    global _max_input_len, _block_pii_input, _redact_pii_output

    if _compiled_injection is not None:
        return  # already initialised

    try:
        from src.config import get_config  # local import to avoid circular

        cfg = get_config().guardrails
        max_len = cfg.max_input_length
        block_pii = cfg.block_pii_input
        redact_pii = cfg.redact_pii_output
        custom_in = [p.pattern for p in cfg.custom_input_patterns]
        custom_out = [p.pattern for p in cfg.custom_output_patterns]
    except Exception:  # noqa: BLE001
        # Config unavailable – fall back to env vars / defaults
        max_len = int(os.getenv("GUARDRAIL_MAX_INPUT_LEN", "4096"))
        block_pii = os.getenv("GUARDRAIL_BLOCK_PII_INPUT", "true").lower() == "true"
        redact_pii = os.getenv("GUARDRAIL_REDACT_PII_OUTPUT", "true").lower() == "true"
        custom_in = []
        custom_out = []

    # Env vars can override config values
    if "GUARDRAIL_MAX_INPUT_LEN" in os.environ:
        max_len = int(os.environ["GUARDRAIL_MAX_INPUT_LEN"])
    if "GUARDRAIL_BLOCK_PII_INPUT" in os.environ:
        block_pii = os.environ["GUARDRAIL_BLOCK_PII_INPUT"].lower() == "true"
    if "GUARDRAIL_REDACT_PII_OUTPUT" in os.environ:
        redact_pii = os.environ["GUARDRAIL_REDACT_PII_OUTPUT"].lower() == "true"

    _max_input_len = max_len
    _block_pii_input = block_pii
    _redact_pii_output = redact_pii

    # Compile injection patterns: built-ins + custom input patterns
    injection_strs = _BUILTIN_INJECTION_PATTERNS + custom_in
    injection_compiled: List[re.Pattern] = []
    for raw in injection_strs:
        try:
            injection_compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Skipping invalid guardrail input pattern %r: %s", raw, exc)
    _compiled_injection = injection_compiled

    # Compile output block patterns: built-ins + custom output patterns
    output_strs = _BUILTIN_OUTPUT_PATTERNS + custom_out
    output_compiled: List[re.Pattern] = []
    for raw in output_strs:
        try:
            output_compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Skipping invalid guardrail output pattern %r: %s", raw, exc)
    _compiled_output = output_compiled

    logger.info(
        "Guardrails initialised: max_input=%d, block_pii=%s, redact_pii=%s, "
        "input_patterns=%d, output_patterns=%d",
        _max_input_len,
        _block_pii_input,
        _redact_pii_output,
        len(_compiled_injection),
        len(_compiled_output),
    )


# ---------------------------------------------------------------------------
# Core check functions
# ---------------------------------------------------------------------------


def _check_injection(text: str) -> str | None:
    """Return the matched snippet if any injection pattern fires, else None."""
    _load_config_patterns()
    for pattern in _compiled_injection or []:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def _contains_pii(text: str) -> bool:
    return any(p.search(text) for p, _ in _PII_PATTERNS)


def _redact_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _check_output_blocklist(text: str) -> str | None:
    """Return the matched snippet if any output block pattern fires, else None."""
    _load_config_patterns()
    for pattern in _compiled_output or []:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class GuardrailsMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware enforcing input/output guardrails on /chat.

    All other paths pass through unmodified.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method == "POST" and request.url.path == "/chat":
            return await self._guard_chat(request, call_next)
        return await call_next(request)

    async def _guard_chat(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Ensure patterns are loaded (lazy, thread-safe via GIL for CPython)
        _load_config_patterns()

        # Buffer body so downstream handler can re-read it
        body_bytes = await request.body()

        async def _replay_receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request = Request(request.scope, _replay_receive)

        try:
            body = json.loads(body_bytes)
            message: str = body.get("message", "")
        except (json.JSONDecodeError, AttributeError):
            message = ""

        # ── INPUT CHECKS ──────────────────────────────────────────────────

        # 1. Length
        max_len = _max_input_len or 4096
        if len(message) > max_len:
            logger.warning("Guardrail: input too long (%d chars)", len(message))
            return _blocked(
                "input_too_long",
                f"Message exceeds maximum allowed length of {max_len} characters.",
            )

        # 2. Injection / jailbreak + custom input patterns
        matched = _check_injection(message)
        if matched:
            logger.warning("Guardrail: injection detected: %r", matched)
            return _blocked("prompt_injection", "Message contains disallowed content.")

        # 3. PII in input
        if _block_pii_input and _contains_pii(message):
            logger.warning("Guardrail: PII detected in input")
            return _blocked(
                "pii_detected",
                "Message appears to contain personal information (PII). "
                "Please remove it before sending.",
            )

        # ── FORWARD TO AGENT ─────────────────────────────────────────────
        response = await call_next(request)

        # ── OUTPUT CHECKS ─────────────────────────────────────────────────
        if response.status_code == 200 and "application/json" in response.headers.get(
            "content-type", ""
        ):
            raw = b""
            async for chunk in response.body_iterator:
                raw += chunk

            try:
                payload = json.loads(raw)
                reply: str = payload.get("reply", "")

                # 4. Output blocklist + custom output patterns
                matched_out = _check_output_blocklist(reply)
                if matched_out:
                    logger.warning("Guardrail: blocked output phrase: %r", matched_out)
                    payload["reply"] = "I'm sorry, I can't provide that information."

                # 5. PII redaction
                elif _redact_pii_output:
                    redacted = _redact_pii(reply)
                    if redacted != reply:
                        logger.info("Guardrail: PII redacted from output")
                    payload["reply"] = redacted

                raw = json.dumps(payload).encode()
            except (json.JSONDecodeError, AttributeError):
                pass

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
