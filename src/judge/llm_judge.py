"""judge/llm_judge.py - LLM-as-a-Judge quality evaluator.

Scores each final agent response against the criteria defined in
``agent_config.yaml``.  Two scorer backends are supported:

use_agent_llm: true  (default)
    Re-uses the same LangChain ``BaseChatModel`` already instantiated for the
    agent, keeping infrastructure simple and avoiding extra credentials.

use_agent_llm: false  (remote endpoint)
    POSTs ``{"question": ..., "response": ...}`` to ``judge.endpoint``
    and expects a JSON response ``{"score": float, "reason": str}``.

Graph integration
-----------------
The judge node runs *after* the LLM produces its final answer.  If the
score is below ``judge.threshold`` the graph routes back to the LLM node for
a retry (up to ``judge.max_retries``).  After max retries the last answer is
returned regardless, to avoid infinite loops.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import JudgeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    score: float          # 0.0 – 1.0
    reason: str
    passed: bool          # score >= config.threshold

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[Judge {status}] score={self.score:.2f} – {self.reason}"


# ---------------------------------------------------------------------------
# LLM-based scorer
# ---------------------------------------------------------------------------

_SCORE_EXTRACTION = re.compile(r'\{\s*"score"\s*:\s*([0-9.]+)', re.IGNORECASE)


async def _score_with_llm(
    llm: BaseChatModel,
    question: str,
    response: str,
    config: JudgeConfig,
) -> JudgeResult:
    """Ask the LLM to score its own (or another model's) response."""
    judge_prompt = (
        f"{config.criteria}\n\n"
        f"User question: {question}\n\n"
        f"AI response:\n{response}"
    )
    messages = [
        SystemMessage(content="You are a strict, objective AI quality evaluator."),
        HumanMessage(content=judge_prompt),
    ]
    try:
        reply = await llm.ainvoke(messages)
        raw_text: str = reply.content if isinstance(reply.content, str) else str(reply.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge LLM call failed: %s", exc)
        return JudgeResult(score=1.0, reason="Judge unavailable – assuming pass", passed=True)

    return _parse_judge_response(raw_text, config.threshold)


def _parse_judge_response(raw: str, threshold: float) -> JudgeResult:
    """Extract ``{"score": float, "reason": str}`` from the LLM reply."""
    # Try strict JSON block first
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("score", 1.0))
            reason = str(parsed.get("reason", ""))
            return JudgeResult(score=score, reason=reason, passed=score >= threshold)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex extraction
    m = _SCORE_EXTRACTION.search(raw)
    if m:
        score = float(m.group(1))
        return JudgeResult(score=score, reason=raw[:200], passed=score >= threshold)

    # Cannot parse → assume pass so we don't block the agent
    logger.warning("Judge could not parse score from response: %r", raw[:300])
    return JudgeResult(score=1.0, reason="Unparseable judge response – assuming pass", passed=True)


# ---------------------------------------------------------------------------
# Remote HTTP scorer
# ---------------------------------------------------------------------------


async def _score_with_endpoint(
    endpoint: str,
    question: str,
    response: str,
    config: JudgeConfig,
) -> JudgeResult:
    payload = {
        "question": question,
        "response": response,
        "criteria": config.criteria,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        score = float(data.get("score", 1.0))
        reason = str(data.get("reason", ""))
        return JudgeResult(score=score, reason=reason, passed=score >= config.threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Remote judge endpoint error: %s", exc)
        return JudgeResult(score=1.0, reason="Judge endpoint unavailable – assuming pass", passed=True)


# ---------------------------------------------------------------------------
# Public evaluate function
# ---------------------------------------------------------------------------


async def evaluate_response(
    question: str,
    response: str,
    config: JudgeConfig,
    llm: Optional[BaseChatModel] = None,
) -> JudgeResult:
    """Score *response* for quality against *question*.

    Args:
        question:  The original user question.
        response:  The agent's final text reply to score.
        config:    :class:`~src.config.JudgeConfig` from ``agent_config.yaml``.
        llm:       The agent's LLM instance.  Required when
                   ``config.use_agent_llm`` is ``True``.

    Returns:
        A :class:`JudgeResult` with ``score``, ``reason``, and ``passed``.
    """
    if not config.enabled:
        return JudgeResult(score=1.0, reason="Judge disabled", passed=True)

    if not config.check_output:
        return JudgeResult(score=1.0, reason="Output checking disabled", passed=True)

    if config.endpoint and not config.use_agent_llm:
        return await _score_with_endpoint(config.endpoint, question, response, config)

    if llm is not None:
        return await _score_with_llm(llm, question, response, config)

    logger.warning("Judge enabled but no LLM provided and no endpoint configured – skipping")
    return JudgeResult(score=1.0, reason="No judge backend available", passed=True)
