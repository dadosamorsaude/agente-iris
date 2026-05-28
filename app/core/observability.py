import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable
from app.core.config import settings

logger = logging.getLogger(__name__)

def configure_langfuse_env() -> None:
    """
    (Legacy) Agora configura LangSmith.
    Se as chaves LangSmith estiverem disponíveis, habilita o tracing.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "projeto-iris")
        logger.info("LangSmith tracing configurado.")
    else:
        logger.info("LANGCHAIN_API_KEY não encontrada, tracing desativado.")


def get_langfuse_callbacks() -> list[Any]:
    """(Legacy) Retorna callbacks vazios. LangSmith opera nativamente via env vars."""
    return []


def flush_langfuse() -> None:
    """(Legacy) LangSmith faz flush automático ou no exit, mantido para compatibilidade."""
    pass


@asynccontextmanager
async def langfuse_lifespan():
    """Context manager para integração com o lifespan do FastAPI."""
    yield


def traceable(func: Callable | None = None, *, name: str | None = None, as_type: str | None = None, **kwargs: Any):
    """
    LangSmith traceable decorator wrapper.
    Se langsmith estiver instalado, usa o @traceable real.
    """
    try:
        from langsmith import traceable as langsmith_traceable
        return langsmith_traceable(name=name, run_type=as_type, **kwargs)(func) if func else langsmith_traceable(name=name, run_type=as_type, **kwargs)
    except ImportError:
        # Fallback se langsmith não estiver instalado
        def decorator(target: Callable):
            from functools import wraps
            @wraps(target)
            def wrapper(*args: Any, **inner_kwargs: Any):
                return target(*args, **inner_kwargs)
            return wrapper
        if func is not None:
            return decorator(func)
        return decorator
