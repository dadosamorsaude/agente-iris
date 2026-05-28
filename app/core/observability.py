import os
from collections.abc import Callable
from functools import lru_cache
from functools import wraps
from typing import Any

from app.core.config import settings


def _configure_langfuse_env() -> None:
    if settings.LANGFUSE_PUBLIC_KEY:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
    if settings.LANGFUSE_SECRET_KEY:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
    if settings.LANGFUSE_BASE_URL:
        os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_BASE_URL)


_configure_langfuse_env()

try:
    from langfuse import get_client as _langfuse_get_client
    from langfuse import observe as _langfuse_observe
except Exception:
    _langfuse_get_client = None
    _langfuse_observe = None


@lru_cache(maxsize=1)
def get_langfuse_callbacks() -> list[Any]:
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        return []

    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return []

    return [CallbackHandler()]


def flush_langfuse() -> None:
    if _langfuse_get_client is None:
        return

    try:
        _langfuse_get_client().flush()
    except Exception:
        return


def traceable(func: Callable | None = None, *, name: str | None = None, as_type: str | None = None, **kwargs: Any):
    """
    Langfuse-backed tracing decorator with the old local call style.

    Keeps the old call styles working:
    - @traceable
    - @traceable(name="tool_name")
    """
    observe_kwargs = dict(kwargs)
    if name:
        observe_kwargs["name"] = name
    if as_type:
        observe_kwargs["as_type"] = as_type

    def decorator(target: Callable):
        if _langfuse_observe is not None:
            return _langfuse_observe(**observe_kwargs)(target)

        @wraps(target)
        def wrapper(*args: Any, **inner_kwargs: Any):
            return target(*args, **inner_kwargs)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
