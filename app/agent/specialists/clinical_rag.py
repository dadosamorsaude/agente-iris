import json
import logging

from app.core.observability import get_langsmith_callbacks, traceable
from app.services.mcp_client import rag_results_context
from app.services.llm import get_chat_model_openai
from app.core.config import settings

logger = logging.getLogger(__name__)


@traceable(name="clinical_rag_expert", as_type="chain")
async def clinical_rag_expert(query: str) -> str:
    logger.info(f"Clinical RAG Expert consultando RAG Catarata via MCP para query: '{query}'")

    try:
        from app.services.mcp_client import invoke_mcp_tool

        # 1. Busca Treinamento e Vocabulário Catarata no MCP concorrentemente
        task_treinamento = invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "catarata",
                "k": 4
            }
        )
        task_vocabulario = invoke_mcp_tool(
            "search_rag_tool",
            {
                "query": query,
                "agent_id": settings.AGENT_ID,
                "namespace_key": "vocabulario",
                "k": 4
            }
        )
        
        response_treinamento, response_vocabulario = await asyncio.gather(
            task_treinamento, task_vocabulario
        )
        
        formatted_treinamento = ""
        if isinstance(response_treinamento, list):
            formatted_treinamento = "".join(item.text if hasattr(item, "text") else str(item) for item in response_treinamento)
        elif isinstance(response_treinamento, str):
            formatted_treinamento = response_treinamento

        formatted_vocabulario = ""
        if isinstance(response_vocabulario, list):
            formatted_vocabulario = "".join(item.text if hasattr(item, "text") else str(item) for item in response_vocabulario)
        elif isinstance(response_vocabulario, str):
            formatted_vocabulario = response_vocabulario

        # Registra dados no contexto para uso do avaliador
        captured = rag_results_context.get([])
        rag_results_context.set(
            captured + [
                {
                    "source": "Treinamento IA Catarata (MCP)",
                    "namespace": "treinamento_ia_catarata",
                    "query": query,
                    "results": formatted_treinamento,
                },
                {
                    "source": "Vocabulário Expandido Catarata (MCP)",
                    "namespace": "catarata_vocabulario_expandido",
                    "query": query,
                    "results": formatted_vocabulario,
                }
            ]
        )

    except Exception as e:
        logger.error(f"Erro ao consultar RAG via MCP no RAG Expert: {e}")
        return "Nenhuma diretriz clínica disponível no momento devido a erro técnico."

    if not formatted_treinamento.strip() and not formatted_vocabulario.strip():
        return "Nenhuma diretriz clínica específica encontrada para esta consulta de cirurgia de catarata."

    llm = get_chat_model_openai(temperature=0.0, model=settings.MODEL_NAME_SQL)

    system_prompt = (
        "Você é o RAG Blueprint do Iris Catarata.\n\n"
        "Função:\n"
        "Recuperar regras de classificação de cirurgia de catarata e retornar JSON estruturado para uso do Agente Orquestrador.\n\n"
        "Fontes:\n"
        "1. treinamento_ia_catarata (regras, scoring, limiares, output esperado - consulte PRIMEIRO)\n"
        "2. catarata_vocabulario_expandido (termos, siglas, CIDs, abreviações - complementar)\n\n"
        "Regras:\n"
        "- Não responda ao usuário.\n"
        "- Não gere SQL.\n"
        "- Não invente regras.\n"
        "- Use SOMENTE o conteúdo recuperado.\n"
        "- Em conflito, prevalece treinamento_ia_catarata.\n"
        "- Procure obrigatoriamente:\n"
        "  1. regras de classificação;\n"
        "  2. scoring;\n"
        "  3. limiares;\n"
        "  4. Output esperado;\n"
        "  5. modelo de extração (orienta o output final do orquestrador, não gera SQL).\n"
        "- Omita campos sem informação.\n"
        "- Não retorne null.\n"
        "- Não retorne string vazia.\n"
        "- Não retorne markdown.\n\n"
        "Retorne SOMENTE JSON válido no formato:\n"
        "{\n"
        "  \"rag_context\": \"resumo das regras clínicas aplicáveis em até 2000 caracteres\",\n"
        "  \"intencao\": \"classificacao|contagem|listagem|distribuicao|relatorio\",\n"
        "  \"procedimento_alvo\": \"cirurgia de catarata\",\n"
        "  \"termos\": {\n"
        "    \"positivos\": [],\n"
        "    \"provaveis\": [],\n"
        "    \"negativos\": [],\n"
        "    \"pos_operatorios\": [],\n"
        "    \"siglas\": [],\n"
        "    \"cids\": []\n"
        "  },\n"
        "  \"scoring\": {\n"
        "    \"regras\": [],\n"
        "    \"limiares\": {}\n"
        "  },\n"
        "  \"output_final_orquestrador\": {\n"
        "    \"found\": true,\n"
        "    \"fonte\": \"treinamento_ia_catarata\",\n"
        "    \"objetivo\": \"\",\n"
        "    \"unidade_analise\": \"\",\n"
        "    \"campos_que_devem_aparecer_na_resposta\": [],\n"
        "    \"evidencias_que_devem_ser_explicadas\": [],\n"
        "    \"classificacoes_que_devem_ser_apresentadas\": [],\n"
        "    \"metricas_que_devem_ser_resumidas\": [],\n"
        "    \"criterios_de_qualidade_da_resposta\": [],\n"
        "    \"instrucao_para_orquestrador\": \"\"\n"
        "  }\n"
        "}"
    )

    user_prompt = (
        f"Pergunta do Usuário: '{query}'\n\n"
        f"--- Trechos de Treinamento IA Catarata ---\n{formatted_treinamento}\n\n"
        f"--- Trechos de Vocabulário Expandido Catarata ---\n{formatted_vocabulario}\n\n"
        "Consolide o JSON estruturado:"
    )

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            config={"callbacks": get_langsmith_callbacks()},
        )
        rag_context = response.content.strip()
        logger.info("Clinical RAG Expert consolidou com sucesso o JSON estruturado do RAG.")
        return rag_context
    except Exception as e:
        logger.error(f"Erro ao sintetizar regras clínicas no RAG Expert: {e}")
        fallback = {
            "rag_context": (formatted_treinamento + "\n" + formatted_vocabulario)[:2000],
            "intencao": "relatorio",
            "procedimento_alvo": "cirurgia de catarata",
            "termos": {"positivos": [], "provaveis": [], "negativos": [], "pos_operatorios": [], "siglas": [], "cids": []},
            "scoring": {"regras": [], "limiares": {}}
        }
        return json.dumps(fallback, ensure_ascii=False)
