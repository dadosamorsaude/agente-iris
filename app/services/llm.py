"""
Construtores de modelos de chat — com cache por (model, temperature).

LangChain reconstrói o cliente HTTP a cada instância, então cacheamos as
combinações usadas para reutilizar conexões TLS e reduzir overhead de boot.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.core.config import settings
from app.core.observability import get_langsmith_callbacks


@lru_cache(maxsize=8)
def _build_openai(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.HTTP_TIMEOUT,
        max_retries=settings.HTTP_MAX_RETRIES,
        callbacks=get_langsmith_callbacks(),
    )


@lru_cache(maxsize=4)
def _build_claude(model: str, temperature: float) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=settings.HTTP_TIMEOUT,
        max_retries=6,
        callbacks=get_langsmith_callbacks(),
    )


def get_chat_model_openai(temperature: float = None, model: str = None) -> ChatOpenAI:
    return _build_openai(
        model=model if model is not None else settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
    )


def get_chat_model_claude(temperature: float = None, model: str = None) -> ChatAnthropic:
    return _build_claude(
        model=model if model is not None else settings.MODEL_CLAUDE,
        temperature=temperature if temperature is not None else settings.TEMPERATURE_CLAUDE,
    )
