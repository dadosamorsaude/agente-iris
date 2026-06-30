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

from app.core.config import settings

logger = logging.getLogger(__name__)


# Busca semântica delegada via MCP server


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
    logger.info(f"Ferramenta Prontuario Search executando via MCP (async): {query}")

    try:
        from app.services.mcp_client import invoke_mcp_tool
        response_obj = await invoke_mcp_tool(
            "search_similar_records_tool",
            {"query": query, "agent_id": settings.AGENT_ID, "top_k": top_k}
        )

        # Processamento robusto do retorno do MCP
        raw_text = ""
        if isinstance(response_obj, list):
            parts = []
            for item in response_obj:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            raw_text = "".join(parts)
        elif isinstance(response_obj, str):
            raw_text = response_obj
        else:
            raw_text = str(response_obj)

        return raw_text

    except Exception as e:
        logger.error(f"Erro em search_similar_records via MCP: {e}")
        return json.dumps({
            "ids_atendimento": [],
            "total_encontrados": 0,
            "sugestao_sql_filter": "",
            "mensagem": f"Erro na busca semântica via MCP: {str(e)}. Prossiga com SQL normalmente.",
        })
