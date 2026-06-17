"""
Singletons de clientes externos.

Por que isto existe
-------------------
No Render plano free temos 512 MB de RAM e CPU compartilhada. Criar um
`OpenAI()`, `Pinecone()`, `httpx.AsyncClient()` ou `OpenAIEmbeddings()` por
request reabre TLS, recarrega tiktoken/gRPC e desperdiça memória. Aqui
mantemos um singleton por processo com timeouts e retries explícitos.

Como usar
---------
    from app.core.clients import (
        openai_async,
        pinecone,
        pinecone_index,
        embeddings_3_large,
        embeddings_3_small_1024,
        supabase_request,
        supabase_headers,
    )

Todas as funções são preguiçosas (instanciam só na primeira chamada).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import httpx
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from app.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI (Async)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def openai_async() -> AsyncOpenAI:
    """AsyncOpenAI compartilhado, com timeout e retries do settings."""
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.HTTP_TIMEOUT,
        max_retries=settings.HTTP_MAX_RETRIES,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings (LangChain wrapper sobre OpenAI)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def embeddings_3_large() -> OpenAIEmbeddings:
    """text-embedding-3-large @ 3072 dims (RAG + busca de prontuários)."""
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-large",
        dimensions=3072,
        timeout=settings.HTTP_TIMEOUT,
        max_retries=settings.HTTP_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def embeddings_3_small_1024() -> OpenAIEmbeddings:
    """text-embedding-3-small @ 1024 dims (semantic cache)."""
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-small",
        dimensions=1024,
        timeout=settings.HTTP_TIMEOUT,
        max_retries=settings.HTTP_MAX_RETRIES,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pinecone
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def pinecone() -> Optional[Pinecone]:
    """Cliente Pinecone (None se PINECONE_API_KEY não estiver configurada)."""
    if not settings.PINECONE_API_KEY:
        return None
    return Pinecone(api_key=settings.PINECONE_API_KEY)


@lru_cache(maxsize=4)
def pinecone_index(name: str):
    """Retorna o handler do índice Pinecone, cacheado por nome."""
    pc = pinecone()
    if pc is None:
        raise RuntimeError("PINECONE_API_KEY não configurada.")
    return pc.Index(name)


# ──────────────────────────────────────────────────────────────────────────────
# Supabase REST
# ──────────────────────────────────────────────────────────────────────────────

def supabase_headers() -> dict:
    """Headers padrão para chamadas REST ao Supabase (vazio se sem API key)."""
    if not settings.DATABASE_API_KEY:
        return {}
    return {
        "apikey": settings.DATABASE_API_KEY,
        "Authorization": f"Bearer {settings.DATABASE_API_KEY}",
        "Content-Type": "application/json",
    }


@lru_cache(maxsize=1)
def supabase_http() -> Optional[httpx.AsyncClient]:
    """
    httpx.AsyncClient configurado para Supabase REST.
    Retorna None se Supabase não estiver configurado.
    """
    base_url = settings.supabase_rest_url
    if not base_url or not settings.DATABASE_API_KEY:
        return None
    return httpx.AsyncClient(
        base_url=base_url,
        headers=supabase_headers(),
        timeout=10.0,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


async def supabase_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json_body: dict | list | None = None,
    extra_headers: dict | None = None,
) -> httpx.Response | None:
    """
    Wrapper único para chamadas REST ao Supabase. Retorna None se Supabase
    não estiver configurado. Erros HTTP são propagados (chame `raise_for_status`
    se quiser falhar; aqui retornamos a Response para o caller decidir).
    """
    client = supabase_http()
    if client is None:
        return None

    headers = None
    if extra_headers:
        headers = {**supabase_headers(), **extra_headers}

    return await client.request(
        method, path, params=params, json=json_body, headers=headers,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shutdown helper (chamado pelo lifespan do FastAPI)
# ──────────────────────────────────────────────────────────────────────────────

async def aclose_clients() -> None:
    """Fecha conexões pendentes — usar no shutdown do FastAPI."""
    client = supabase_http.cache_info()
    if client.currsize > 0:
        http = supabase_http()
        if http is not None:
            try:
                await http.aclose()
            except Exception as e:
                logger.warning(f"Falha ao fechar supabase_http: {e}")
