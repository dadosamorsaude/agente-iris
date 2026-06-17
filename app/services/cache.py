"""
Cache semântico via Pinecone — usado pelo iris_chat.

Lazy init: o cliente Pinecone e o embeddings só são construídos quando
o cache é REALMENTE invocado (não no import do módulo). Isso evita
sobrecarga de boot quando o cache está desligado.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from app.core.clients import embeddings_3_small_1024, pinecone_index
from app.core.config import settings

logger = logging.getLogger(__name__)


class PineconeSemanticCache:
    def __init__(self) -> None:
        self.enabled = bool(
            settings.PINECONE_API_KEY and settings.PINECONE_INDEX_CACHE
        )
        self.namespace = settings.PINECONE_CACHE_NAMESPACE

    def _index(self):
        return pinecone_index(settings.PINECONE_INDEX_CACHE)

    def _embeddings(self):
        return embeddings_3_small_1024()

    async def get(self, query: str, threshold: float = 0.95) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            vector = await self._embeddings().aembed_query(query)
            result = self._index().query(
                vector=vector,
                top_k=1,
                include_metadata=True,
                namespace=self.namespace,
            )
            if result.matches and result.matches[0].score >= threshold:
                logger.info(
                    f"Semantic Cache HIT! (Similaridade: {result.matches[0].score:.4f})"
                )
                return result.matches[0].metadata
            logger.info("Semantic Cache MISS")
            return None
        except Exception as e:
            logger.error(f"Erro no cache semantico (get): {e}")
            return None

    async def set(
        self,
        query: str,
        response: str,
        athena_data: list | None = None,
        rag_data: list | None = None,
    ) -> None:
        if not self.enabled:
            return
        if not athena_data:
            logger.info("Cache semantico: skipping (sem dados do Athena)")
            return
        try:
            vector = await self._embeddings().aembed_query(query)
            metadata = {
                "query": query,
                "response": response,
                "athena_data": json.dumps(athena_data or []),
                "rag_data": json.dumps(rag_data or []),
            }
            self._index().upsert(
                vectors=[{
                    "id": str(uuid.uuid4()),
                    "values": vector,
                    "metadata": metadata,
                }],
                namespace=self.namespace,
            )
            logger.info("Semantic Cache atualizado.")
        except Exception as e:
            logger.error(f"Erro no cache semantico (set): {e}")


# Instância global (cliente/embeddings só são tocados quando enabled=True
# e na primeira chamada — lazy via app.core.clients).
semantic_cache = PineconeSemanticCache()
