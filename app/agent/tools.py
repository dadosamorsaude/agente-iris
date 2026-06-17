from datetime import date
from langchain_core.tools import tool

from app.agent.specialists.clinical_rag import clinical_rag_expert
from app.agent.specialists.sql_analyst import sql_analyst_expert
from app.services.learning import load_curated_lessons
from app.tools.rag import rag_results_context
from app.tools.athena import athena_results_context
from app.tools.prontuario_search import search_similar_records

@tool
async def fetch_clinical_guidelines(query: str) -> str:
    """Busca as diretrizes clínicas e de negócio (RAG) da régua de catarata. 
    Use esta ferramenta SEMPRE que a pergunta do usuário precisar de contexto clínico para ser respondida.
    """
    return await clinical_rag_expert(query)

@tool
async def fetch_curated_lessons() -> str:
    """Busca aprendizados históricos (memória) para evitar erros recorrentes.
    Use esta ferramenta para verificar se existe algum padrão de erro conhecido na mesma intenção do usuário.
    """
    return await load_curated_lessons()

@tool
async def analyze_and_execute_sql(query: str, rag_context: str) -> dict:
    """Gera, valida e executa uma consulta SQL no banco de dados para buscar atendimentos de catarata.
    Sempre chame 'fetch_clinical_guidelines' ANTES de usar esta ferramenta, e passe o resultado no parâmetro 'rag_context'.
    Retorna os dados resultantes da consulta (agregados ou brutos).
    """
    hoje = date.today().isoformat()
    result = await sql_analyst_expert(
        query=query,
        rag_context=rag_context,
        output_mode=None,
        sample_size=5,
        hoje=hoje
    )
    return result

tools = [fetch_clinical_guidelines, fetch_curated_lessons, analyze_and_execute_sql, search_similar_records]
