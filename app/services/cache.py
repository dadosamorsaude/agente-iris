from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class PineconeSemanticCache:
    def __init__(self):
        self.enabled = bool(settings.PINECONE_API_KEY and getattr(settings, 'PINECONE_INDEX_CACHE', None))
        if self.enabled:
            try:
                self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                self.index = self.pc.Index(settings.PINECONE_INDEX_CACHE)
                self.embeddings = OpenAIEmbeddings(
                    api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small",
                    dimensions=1024
                )
            except Exception as e:
                logger.error(f"Erro ao inicializar PineconeSemanticCache: {e}")
                self.enabled = False
            
    async def get(self, query: str, threshold: float = 0.95):
        if not self.enabled: return None
        try:
            vector = await self.embeddings.aembed_query(query)
            result = self.index.query(vector=vector, top_k=1, include_metadata=True)
            
            if result.matches and result.matches[0].score >= threshold:
                score_str = f"{result.matches[0].score:.4f}"
                logger.info(f"🟢 Semantic Cache HIT! (Similaridade: {score_str})")
                return result.matches[0].metadata
            
            logger.info("🟡 Semantic Cache MISS")
            return None
        except Exception as e:
            logger.error(f"Erro no cache semantico (get): {e}")
            return None
            
    async def set(self, query: str, response: str, athena_data: list = None, rag_data: list = None):
        if not self.enabled: return
        try:
            vector = await self.embeddings.aembed_query(query)
            metadata = {
                "query": query, 
                "response": response,
                "athena_data": json.dumps(athena_data or []),
                "rag_data": json.dumps(rag_data or [])
            }
            self.index.upsert(vectors=[{
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": metadata
            }])
            logger.info("🔵 Semantic Cache atualizado com a nova resposta e metadados.")
        except Exception as e:
            logger.error(f"Erro no cache semantico (set): {e}")

# Instância global do cache
semantic_cache = PineconeSemanticCache()
