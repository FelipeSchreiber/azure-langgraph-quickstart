"""memory/checkpoint.py - LangGraph checkpointer factory.

Reads the ``memory`` block from ``agent_config.yaml`` and returns the
appropriate LangGraph checkpointer.

Supported backends
------------------
none       No checkpointer (default).  Each request is stateless.
            Conversation history is not persisted across requests.

memory     :class:`langgraph.checkpoint.memory.MemorySaver`
            Pure in-process dict store.  Lost on restart.  Good for
            local development and unit tests.

sqlite     :class:`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
            Single-file SQLite database.  Survives restarts.
            Set ``memory.db_path`` in agent_config.yaml.

postgres   :class:`langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`
            Full PostgreSQL backend.  Suitable for multi-replica production
            deployments.  Set the DSN via ``memory.connection_env_var``.

Returns ``None`` when ``type`` is ``none`` (or the ``memory`` section is
omitted from ``agent_config.yaml``).  Passing ``None`` to
``StateGraph.compile(checkpointer=None)`` is valid — LangGraph simply
runs without persistence.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import MemoryConfig

logger = logging.getLogger(__name__)


async def build_checkpointer(config: MemoryConfig) -> Any:
    """Build and return the appropriate LangGraph checkpointer, or ``None``.

    When ``config.type`` is ``"none"`` (or the ``memory`` section is absent
    from ``agent_config.yaml``), returns ``None`` — the graph will run without
    any persistence and every request will be stateless.

    Args:
        config: :class:`~src.config.MemoryConfig` parsed from ``agent_config.yaml``.

    Returns:
        A checkpointer instance, or ``None`` when persistence is disabled.

    Raises:
        ImportError: When the required optional dependency is not installed.
        ValueError:  When an unsupported ``type`` is specified.
    """
    ctype = config.type.lower()

    # ── Disabled (no persistence) ──────────────────────────────────────────
    if ctype == "none":
        logger.info("Checkpointer: disabled (stateless mode)")
        return None

    # ── In-memory ──────────────────────────────────────────────────────────
    if ctype == "memory":
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-untyped]
        logger.info("Checkpointer: MemorySaver (in-process, ephemeral)")
        return MemorySaver()

    # ── SQLite ─────────────────────────────────────────────────────────────
    if ctype == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "SQLite checkpointer requires 'langgraph-checkpoint-sqlite'. "
                "Install it with: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = config.db_path or ":memory:"
        logger.info("Checkpointer: AsyncSqliteSaver (path=%s)", db_path)
        return AsyncSqliteSaver.from_conn_string(db_path)

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    if ctype == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL checkpointer requires 'langgraph-checkpoint-postgres'. "
                "Install it with: pip install langgraph-checkpoint-postgres"
            ) from exc

        conn_str = config.resolve_connection_string()
        if not conn_str:
            raise ValueError(
                f"memory.connection_env_var '{config.connection_env_var}' is not set "
                "or resolves to an empty string. "
                "Export the PostgreSQL DSN (e.g. "
                "CHECKPOINT_DB_URL=postgresql+psycopg://user:pass@host/db) "
                "before starting the agent."
            )

        logger.info("Checkpointer: AsyncPostgresSaver (table=%s)", config.table_name)
        checkpointer = await AsyncPostgresSaver.from_conn_string(conn_str)
        # Create tables if they don't exist yet
        await checkpointer.setup()
        return checkpointer

    raise ValueError(
        f"Unsupported memory.type '{config.type}'. "
        "Supported values: none | memory | sqlite | postgres"
    )
