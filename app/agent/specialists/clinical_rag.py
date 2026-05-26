import logging
from app.tools.rag import get_retriever, rag_results_context, format_docs
from app.services.llm import get_chat_model_openai

logger = logging.getLogger(__name__)

def clinical_rag_expert(query: str) -> str:
    """
    Especialista clínico em RAG Catarata.
    Consulta o index 'rag-agente-cirurgias', namespace 'catarata_vocabulario_expandido'
    e retorna o rag_context estruturado.
    """
    logger.info(f"Clinical RAG Expert consultando RAG Catarata para query: '{query}'")
    
    retriever = get_retriever("rag-agente-cirurgias", "catarata_vocabulario_expandido", k=5)
    
    if not retriever:
        logger.warning("Pinecone não configurado. RAG Expert indisponível.")
        return "Nenhuma diretriz de catarata disponível no momento."

    docs = retriever.invoke(query)
    
    # Salva no contextvar
    captured = rag_results_context.get([])
    rag_results_context.set(
        captured + [
            {
                "source": "RAG Catarata",
                "namespace": "catarata_vocabulario_expandido",
                "query": query,
                "chunks": [d.page_content for d in docs],
                "metadata": [d.metadata for d in docs],
            }
        ]
    )

    if not docs:
        return "Nenhuma diretriz clínica específica encontrada para esta consulta de cirurgia de catarata."

    # Formata trechos recuperados
    formatted_docs = format_docs(docs)

    # Invoca uma chamada rápida ao LLM para estruturar as regras clínicas obtidas de forma limpa e compreensível
    llm = get_chat_model_openai(temperature=0.0)
    
    system_prompt = (
        "Você é um auditor clínico e especialista em cirurgia de catarata.\n"
        "Com base nos trechos da documentação de treinamento recuperados pelo RAG, "
        "sintetize com extrema fidelidade as regras clínicas, léxicos de detecção, "
        "tabela de scoring, limitações, conceitos ou exemplos relevantes para auditar a pergunta do usuário.\n"
        "Seja sucinto, técnico e responda em português. Não adicione opiniões pessoais, invente regras ou cite documentos que não estão nos trechos."
    )
    
    user_prompt = (
        f"Pergunta do Usuário: '{query}'\n\n"
        f"Trechos de Documentação Técnica:\n{formatted_docs}\n\n"
        "Consolide as regras e termos clínicos aplicáveis:"
    )
    
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        rag_context = response.content.strip()
        logger.info("Clinical RAG Expert sintetizou com sucesso as regras clínicas.")
        return rag_context
    except Exception as e:
        logger.error(f"Erro ao sintetizar regras clínicas no RAG Expert: {e}")
        return formatted_docs
