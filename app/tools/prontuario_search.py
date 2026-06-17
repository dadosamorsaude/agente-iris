"""
Tool de Busca Semântica de Prontuários — Iris

Invocada internamente pelo React Agent para encontrar prontuários
clinicamente similares a uma query clínica derivada internamente da
pergunta do usuário. O usuário nunca vê essa ferramenta diretamente.

Fluxo esperado:
    fetch_clinical_guidelines → search_similar_records → analyze_and_execute_sql
                                        ↑
                    query derivada internamente pelo agente a partir do léxico do RAG
"""

import asyncio
import json
import logging

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from app.core.config import settings

logger = logging.getLogger(__name__)


def _search_pinecone(query_vector: list[float], top_k: int) -> list:
    """Executa busca vetorial síncrona no Pinecone (chamada via asyncio.to_thread)."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_RAG_INDEX)
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=settings.PINECONE_NS_PRONTUARIOS,
    )
    return result.matches


@tool
async def search_similar_records(query: str, top_k: int = 20) -> str:
    """
    Busca prontuários clinicamente similares usando busca vetorial no Pinecone.

    IMPORTANTE: Esta ferramenta é invocada INTERNAMENTE pela Iris.
    O parâmetro 'query' deve ser uma representação clínica derivada
    do léxico do RAG (ex: 'facectomia indicação catarata LIO'),
    NÃO a pergunta literal do usuário.

    Use quando precisar ampliar o recall de casos clínicos com variações
    de terminologia (ex: 'FACO', 'facectomia', 'extração de cristalino',
    erros de digitação, abreviações). Retorna uma lista de id_atendimento
    para usar como filtro adicional WHERE em analyze_and_execute_sql,
    aumentando a cobertura de casos que o regex do SQL poderia perder.
    """
    if not settings.PINECONE_API_KEY:
        logger.warning("search_similar_records: Pinecone não configurado, retornando vazio.")
        return json.dumps({
            "ids_atendimento": [],
            "total_encontrados": 0,
            "sugestao_sql_filter": "",
            "mensagem": "Busca semântica indisponível (Pinecone não configurado).",
        })

    try:
        embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-large",
            dimensions=3072,
        )
        query_vector = await embeddings.aembed_query(query)
        matches = await asyncio.to_thread(_search_pinecone, query_vector, top_k)

        if not matches:
            return json.dumps({
                "ids_atendimento": [],
                "total_encontrados": 0,
                "sugestao_sql_filter": "",
                "mensagem": (
                    "Nenhum prontuário similar encontrado no índice semântico. "
                    "Prossiga com a busca SQL usando regex normalmente."
                ),
            })

        ids = []
        trechos = []
        for match in matches:
            meta = match.metadata or {}
            id_atend = meta.get("id_atendimento")
            if not id_atend:
                continue
            ids.append(id_atend)
            trechos.append({
                "id_atendimento": id_atend,
                "score_similaridade": round(float(match.score), 4),
                "clinica": meta.get("clinica", ""),
                "regional": meta.get("regional", ""),
                "data_atendimento": meta.get("data_atendimento", ""),
                "trecho": meta.get("text", "")[:300],
            })

        sql_filter = f"id_atendimento IN ({', '.join(ids)})" if ids else ""

        result = {
            "ids_atendimento": ids,
            "total_encontrados": len(ids),
            "sugestao_sql_filter": sql_filter,
            "instrucao": (
                "Use 'sugestao_sql_filter' como filtro adicional no SQL para "
                "ampliar o recall de casos clínicos com variações de terminologia. "
                "Combine com os filtros de data e clínica já extraídos da pergunta."
            ),
            "trechos": trechos,
        }

        logger.info(
            f"search_similar_records: {len(ids)} prontuários encontrados "
            f"| query='{query[:60]}'"
        )
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Erro em search_similar_records: {e}")
        return json.dumps({
            "ids_atendimento": [],
            "total_encontrados": 0,
            "sugestao_sql_filter": "",
            "mensagem": f"Erro na busca semântica: {str(e)}. Prossiga com SQL normalmente.",
        })
