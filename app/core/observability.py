import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def configure_langfuse_env() -> None:
    """
    Configura as variáveis de ambiente do Langfuse a partir das settings.
    Deve ser chamada APÓS o load_dotenv(), no startup da aplicação.
    Usa atribuição direta (não setdefault) para garantir que os valores do .env
    sempre prevaleçam sobre qualquer valor pré-existente no ambiente.
    """
    if settings.LANGFUSE_PUBLIC_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    if settings.LANGFUSE_SECRET_KEY:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    if settings.LANGFUSE_BASE_URL:
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL


try:
    from langfuse import get_client as _langfuse_get_client
    from langfuse import observe as _langfuse_observe
    _LANGFUSE_AVAILABLE = True
except Exception:
    _langfuse_get_client = None
    _langfuse_observe = None
    _LANGFUSE_AVAILABLE = False


def get_langfuse_callbacks() -> list[Any]:
    """
    Retorna os callbacks do Langfuse para uso nos modelos LangChain.
    Não usa lru_cache para garantir que o callback sempre reflita o estado
    atual da configuração (evita cache com estado "frio").
    """
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        return []

    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception as exc:
        logger.debug("Langfuse CallbackHandler indisponível: %s", exc)
        return []


def flush_langfuse() -> None:
    """Força o envio de todos os traces pendentes ao Langfuse."""
    if _langfuse_get_client is None:
        return
    try:
        _langfuse_get_client().flush()
        logger.debug("Langfuse flush concluído.")
    except Exception as exc:
        logger.debug("Langfuse flush ignorado: %s", exc)


@asynccontextmanager
async def langfuse_lifespan():
    """
    Context manager para integração com o lifespan do FastAPI.
    Garante que o flush é chamado no shutdown, evitando perda de traces.

    Uso:
        @asynccontextmanager
        async def lifespan(app):
            async with langfuse_lifespan():
                yield
    """
    yield
    flush_langfuse()


def traceable(func: Callable | None = None, *, name: str | None = None, as_type: str | None = None, **kwargs: Any):
    """
    Langfuse-backed tracing decorator.

    Estilos suportados:
    - @traceable
    - @traceable(name="tool_name")
    - @traceable(name="span_name", as_type="span")
    """
    observe_kwargs = dict(kwargs)
    if name:
        observe_kwargs["name"] = name
    if as_type:
        observe_kwargs["as_type"] = as_type

    def decorator(target: Callable):
        if _langfuse_observe is not None:
            return _langfuse_observe(**observe_kwargs)(target)

        from functools import wraps

        @wraps(target)
        def wrapper(*args: Any, **inner_kwargs: Any):
            return target(*args, **inner_kwargs)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
