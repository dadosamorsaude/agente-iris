import asyncio
import json
from contextvars import ContextVar
from typing import Any
import logging
from app.core.config import settings
from langchain_core.tools import tool
from app.core.observability import traceable

logger = logging.getLogger(__name__)

athena_results_context: ContextVar[list] = ContextVar("athena_results", default=[])


def validate_sql(sql: str) -> None:
    """Validates SQL to allow only read-only SELECT queries."""
    sql_upper = sql.upper()
    forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE "]
    if any(token in sql_upper for token in forbidden):
        logger.error(f"Operacao proibida detectada no SQL: {sql}")
        raise ValueError("SQL contem operacao proibida. Apenas SELECT e permitido.")

    if "SELECT *" in sql_upper:
        raise ValueError("SELECT * nao e permitido. Por favor, liste as colunas explicitamente.")



@tool
@traceable(name="query_athena_tool")
async def query_athena_tool(sql: str) -> str:
    """
    Executa consultas SQL no AWS Athena para analise de prontuarios medicos.
    A query deve ser compativel com Presto/Athena.
    O teto de linhas e controlado por settings.ATHENA_MAX_ROWS (default: sem teto).
    """
    try:
        validate_sql(sql)
    except ValueError as e:
        logger.warning(f"SQL invalido rejeitado: {e}")
        return f"Consulta invalida: {str(e)}"

    logger.info(f"Ferramenta Athena executando via MCP (async): {sql}")

    try:
        from app.services.mcp_client import invoke_mcp_tool
        response_obj = await invoke_mcp_tool("query_athena_tool", {"sql": sql, "agent_id": settings.AGENT_ID})

        # Processamento robusto
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

        payload = {}
        try:
            payload = json.loads(raw_text)
        except Exception:
            payload = raw_text

        # Captura os dados brutos no contexto local para uso pelo Agente Avaliador
        results = []
        try:
            if isinstance(payload, dict):
                results = payload.get("rows", [])
                hit_limit = payload.get("row_limit_hit", False)
                captured = athena_results_context.get([])
                athena_results_context.set(
                    captured + [{"sql": sql, "results": results, "row_limit_hit": hit_limit}]
                )
                logger.info(f"Ferramenta Athena: Retornadas {len(results)} linhas via MCP.")
                return json.dumps(results, default=str, ensure_ascii=False)
            elif isinstance(payload, list):
                results = payload
                captured = athena_results_context.get([])
                athena_results_context.set(
                    captured + [{"sql": sql, "results": results, "row_limit_hit": False}]
                )
                logger.info(f"Ferramenta Athena: Retornadas {len(results)} linhas via MCP.")
                return raw_text
        except Exception as parse_err:
            logger.warning(f"Erro ao capturar dados do MCP no athena_results_context: {parse_err}")

        return raw_text

    except Exception as e:
        logger.exception("Erro na ferramenta Athena via MCP")
        return f"Erro ao acessar o banco de dados Athena via MCP: {str(e)}."
