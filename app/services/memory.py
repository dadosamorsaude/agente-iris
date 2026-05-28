import json
import uuid
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from app.core.config import settings
import logging
import asyncpg

logger = logging.getLogger(__name__)

# In-memory fallback (used when DATABASE_URL is not configured)
_memory_store: dict = {}
_tables_created = False


def _to_uuid(session_id: str) -> str:
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, session_id))


async def ensure_tables(pool: asyncpg.Pool) -> None:
    global _tables_created
    if _tables_created:
        return
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                message JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_session
            ON chat_history (session_id)
        """)
    _tables_created = True
    logger.info("Tabela 'chat_history' verificada/criada com sucesso (asyncpg).")


async def get_pool() -> Optional[asyncpg.Pool]:
    if not settings.DATABASE_URL:
        return None
    try:
        pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
        if pool:
            await ensure_tables(pool)
        return pool
    except Exception as e:
        logger.warning(f"Falha ao criar pool asyncpg: {e}")
        return None


_pool: Optional[asyncpg.Pool] = None


async def get_session_history(session_id: str, max_messages: int = 8) -> list[BaseMessage]:
    global _pool
    valid_session_id = _to_uuid(session_id)

    if settings.DATABASE_URL:
        try:
            if _pool is None:
                _pool = await get_pool()

            if _pool:
                async with _pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT message FROM chat_history
                           WHERE session_id = $1
                           ORDER BY created_at ASC
                           LIMIT $2""",
                        valid_session_id, max_messages * 2
                    )
                messages = []
                for row in rows:
                    msg_data = json.loads(row["message"])
                    msg_type = msg_data.get("type", "human")
                    content = msg_data.get("content", "")
                    if msg_type == "ai":
                        messages.append(AIMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=content))
                return messages[-max_messages:] if len(messages) > max_messages else messages
        except Exception as e:
            logger.warning(f"Falha ao buscar histórico do PostgreSQL: {e}")

    # Fallback: in-memory
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    return _memory_store[session_id][-max_messages:]


async def add_message(session_id: str, message: BaseMessage) -> None:
    global _pool
    valid_session_id = _to_uuid(session_id)
    msg_type = "ai" if isinstance(message, AIMessage) else "human"
    msg_data = json.dumps({"type": msg_type, "content": message.content})

    if settings.DATABASE_URL:
        try:
            if _pool is None:
                _pool = await get_pool()
            if _pool:
                async with _pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO chat_history (session_id, message) VALUES ($1, $2::jsonb)",
                        valid_session_id, msg_data
                    )
                return
        except Exception as e:
            logger.warning(f"Falha ao salvar mensagem no PostgreSQL: {e}")

    # Fallback: in-memory
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    _memory_store[session_id].append(message)


async def add_user_message(session_id: str, content: str) -> None:
    await add_message(session_id, HumanMessage(content=content))


async def add_ai_message(session_id: str, content: str) -> None:
    await add_message(session_id, AIMessage(content=content))


async def clear_session_history(session_id: str) -> None:
    global _pool
    valid_session_id = _to_uuid(session_id)

    if settings.DATABASE_URL:
        try:
            if _pool is None:
                _pool = await get_pool()
            if _pool:
                async with _pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM chat_history WHERE session_id = $1",
                        valid_session_id
                    )
                return
        except Exception as e:
            logger.warning(f"Falha ao limpar histórico no PostgreSQL: {e}")

    if session_id in _memory_store:
        del _memory_store[session_id]


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
