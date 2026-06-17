"""
Histórico de sessão (chat history) — persistido em Postgres se DATABASE_URL
estiver configurado; caso contrário, fallback in-memory LIMITADO por sessão
(deque com maxlen) e LRU global por número de sessões. Isso evita OOM no
Render plano free (512 MB) quando o banco não está disponível.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import OrderedDict, deque
from typing import Deque, Optional

import asyncpg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Fallback in-memory: LRU global por sessão, deque por sessão.
# ──────────────────────────────────────────────────────────────────────────────
MEMORY_MAX_SESSIONS = 100        # nº máximo de sessões no fallback
MEMORY_MAX_MESSAGES_PER_SESSION = 20  # nº máximo de mensagens por sessão

_memory_store: "OrderedDict[str, Deque[BaseMessage]]" = OrderedDict()
_tables_created = False
_pool: Optional[asyncpg.Pool] = None


def _memory_get(session_id: str) -> Deque[BaseMessage]:
    """Retorna o deque da sessão, criando-o e atualizando o LRU."""
    if session_id in _memory_store:
        _memory_store.move_to_end(session_id)
        return _memory_store[session_id]
    if len(_memory_store) >= MEMORY_MAX_SESSIONS:
        _memory_store.popitem(last=False)  # remove a mais antiga
    bucket: Deque[BaseMessage] = deque(maxlen=MEMORY_MAX_MESSAGES_PER_SESSION)
    _memory_store[session_id] = bucket
    return bucket


def _to_uuid(session_id: str) -> str:
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, session_id))


# ──────────────────────────────────────────────────────────────────────────────
# Pool Postgres
# ──────────────────────────────────────────────────────────────────────────────


async def ensure_tables(pool: asyncpg.Pool) -> None:
    global _tables_created
    if _tables_created:
        return
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history_iris (
                id SERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                message JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_session_iris
            ON chat_history_iris (session_id)
        """)
    _tables_created = True
    logger.info("Tabela 'chat_history_iris' verificada/criada com sucesso.")


async def get_pool() -> Optional[asyncpg.Pool]:
    if not settings.DATABASE_URL:
        return None
    try:
        pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            timeout=10,           # aquisição
            command_timeout=30,   # por comando
        )
        if pool:
            await ensure_tables(pool)
        return pool
    except Exception as e:
        logger.warning(f"Falha ao criar pool asyncpg: {e}")
        return None


async def _ensure_pool() -> Optional[asyncpg.Pool]:
    global _pool
    if _pool is None and settings.DATABASE_URL:
        _pool = await get_pool()
    return _pool


async def close_pool() -> None:
    """Fechado pelo lifespan do FastAPI no shutdown."""
    global _pool
    if _pool:
        try:
            await _pool.close()
        except Exception as e:
            logger.warning(f"Falha ao fechar pool asyncpg: {e}")
        _pool = None


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────


async def get_session_history(
    session_id: str, max_messages: int = 8
) -> list[BaseMessage]:
    valid_session_id = _to_uuid(session_id)

    pool = await _ensure_pool()
    if pool:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT message FROM chat_history_iris
                       WHERE session_id = $1
                       ORDER BY created_at ASC
                       LIMIT $2""",
                    valid_session_id, max_messages * 2,
                )
            messages: list[BaseMessage] = []
            for row in rows:
                msg_data = json.loads(row["message"])
                msg_type = msg_data.get("type", "human")
                content = msg_data.get("content", "")
                if msg_type == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            return messages[-max_messages:]
        except Exception as e:
            logger.warning(f"Falha ao buscar histórico do PostgreSQL: {e}")

    return list(_memory_get(session_id))[-max_messages:]


async def add_message(session_id: str, message: BaseMessage) -> None:
    valid_session_id = _to_uuid(session_id)
    msg_type = "ai" if isinstance(message, AIMessage) else "human"
    msg_data = json.dumps({"type": msg_type, "content": message.content})

    pool = await _ensure_pool()
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO chat_history_iris (session_id, message) VALUES ($1, $2::jsonb)",
                    valid_session_id, msg_data,
                )
            return
        except Exception as e:
            logger.warning(f"Falha ao salvar mensagem no PostgreSQL: {e}")

    _memory_get(session_id).append(message)


async def add_user_message(session_id: str, content: str) -> None:
    await add_message(session_id, HumanMessage(content=content))


async def add_ai_message(session_id: str, content: str) -> None:
    await add_message(session_id, AIMessage(content=content))
