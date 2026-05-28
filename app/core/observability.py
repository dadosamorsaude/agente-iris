import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable
from app.core.config import settings

logger = logging.getLogger(__name__)


def configure_langsmith() -> None:
    """Configura LangSmith tracing se a chave de API estiver disponível."""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "projeto-iris")
        logger.info("LangSmith tracing configurado.")
    else:
        logger.info("LANGCHAIN_API_KEY não encontrada, tracing desativado.")


def get_langsmith_callbacks() -> list[Any]:
    """LangSmith opera nativamente via env vars, sem callbacks explícitos."""
    return []


def flush_langsmith() -> None:
    """LangSmith faz flush automático. Mantido para compatibilidade."""
    pass


@asynccontextmanager
async def langsmith_lifespan():
    yield


def traceable(func: Callable | None = None, *, name: str | None = None, as_type: str | None = None, **kwargs: Any):
    """LangSmith traceable decorator wrapper."""
    try:
        from langsmith import traceable as langsmith_traceable
        return langsmith_traceable(name=name, run_type=as_type, **kwargs)(func) if func else langsmith_traceable(name=name, run_type=as_type, **kwargs)
    except ImportError:
        def decorator(target: Callable):
            from functools import wraps
            @wraps(target)
            def wrapper(*args: Any, **inner_kwargs: Any):
                return target(*args, **inner_kwargs)
            return wrapper
        if func is not None:
            return decorator(func)
        return decorator
