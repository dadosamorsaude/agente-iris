import logging
import json
from app.core.observability import get_langfuse_callbacks, traceable
from app.tools.rag import get_retriever, rag_results_context, format_docs
from app.services.llm import get_chat_model_openai

logger = logging.getLogger(__name__)

@traceable(name="clinical_rag_expert", as_type="span")
def clinical_rag_expert(query: str) -> str:
    """
    Especialista clínico em RAG Catarata.
    Consulta o index 'rag-agente-cirurgias' em dois namespaces:
    1. 'treinamento_ia_catarata' (Regras, scoring, output esperado - PRIORITÁRIO)
    2. 'catarata_vocabulario_expandido' (Termos, siglas, CIDs - COMPLEMENTAR)
    Retorna o rag_context estruturado como JSON textual para o orquestrador.
    """
    logger.info(f"Clinical RAG Expert consultando RAG Catarata para query: '{query}'")
    
    # 1. Recupera as regras clínicas do RAG prioritário
    retriever_treinamento = get_retriever("rag-agente-cirurgias", "treinamento_ia_catarata", k=4)
    # 2. Recupera o vocabulário complementar
    retriever_vocabulario = get_retriever("rag-agente-cirurgias", "catarata_vocabulario_expandido", k=4)
    
    if not retriever_treinamento or not retriever_vocabulario:
        logger.warning("Pinecone não configurado. RAG Expert indisponível.")
        return "Nenhuma diretriz de catarata disponível no momento."

    docs_treinamento = retriever_treinamento.invoke(query)
    docs_vocabulario = retriever_vocabulario.invoke(query)
    
    # Salva no contextvar para auditoria posterior pelo Judge
    captured = rag_results_context.get([])
    rag_results_context.set(
        captured + [
            {
                "source": "Treinamento IA Catarata",
                "namespace": "treinamento_ia_catarata",
                "query": query,
                "chunks": [d.page_content for d in docs_treinamento],
                "metadata": [d.metadata for d in docs_treinamento],
            },
            {
                "source": "Vocabulário Expandido Catarata",
                "namespace": "catarata_vocabulario_expandido",
                "query": query,
                "chunks": [d.page_content for d in docs_vocabulario],
                "metadata": [d.metadata for d in docs_vocabulario],
            }
        ]
    )

    if not docs_treinamento and not docs_vocabulario:
        return "Nenhuma diretriz clínica específica encontrada para esta consulta de cirurgia de catarata."

    # Formata trechos recuperados de ambas as fontes
    formatted_treinamento = format_docs(docs_treinamento)
    formatted_vocabulario = format_docs(docs_vocabulario)

    # Invoca chamada ao LLM para estruturar o RAG Context em JSON conforme especificado no n8n (rag.json)
    llm = get_chat_model_openai(temperature=0.0)
    
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
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            config={"callbacks": get_langfuse_callbacks()},
        )
        rag_context = response.content.strip()
        logger.info("Clinical RAG Expert consolidou com sucesso o JSON estruturado do RAG.")
        return rag_context
    except Exception as e:
        logger.error(f"Erro ao sintetizar regras clínicas no RAG Expert: {e}")
        # Retorna uma estrutura basica contendo os textos brutos se falhar
        fallback = {
            "rag_context": (formatted_treinamento + "\n" + formatted_vocabulario)[:2000],
            "intencao": "relatorio",
            "procedimento_alvo": "cirurgia de catarata",
            "termos": {"positivos": [], "provaveis": [], "negativos": [], "pos_operatorios": [], "siglas": [], "cids": []},
            "scoring": {"regras": [], "limiares": {}}
        }
        return json.dumps(fallback, ensure_ascii=False)
