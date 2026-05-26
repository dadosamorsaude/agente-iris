import asyncio
from contextvars import ContextVar
from pyathena import connect
from app.core.config import settings
from langchain_core.tools import tool
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)

# Contexto por-task que armazena os dados brutos retornados pelo Athena.
# Permite que o Agente Avaliador acesse os dados sem re-executar queries.
athena_results_context: ContextVar[list] = ContextVar("athena_results", default=[])


def validate_sql(sql: str) -> None:
    """Validates SQL to allow only read-only SELECT queries."""
    sql_upper = sql.upper()
    forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE "]
    if any(token in sql_upper for token in forbidden):
        logger.error(f"Operação proibida detectada no SQL: {sql}")
        raise ValueError("SQL contém operação proibida. Apenas SELECT é permitido.")

    if "SELECT *" in sql_upper:
        raise ValueError("SELECT * não é permitido. Por favor, liste as colunas explicitamente.")


def _execute_athena_query(sql: str):
    """Internal synchronous function to execute the query."""
    conn = None
    cursor = None
    try:
        conn = connect(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.aws_region_clean,
            s3_staging_dir=settings.ATHENA_S3_STAGING_DIR,
            schema_name=settings.ATHENA_DATABASE,
        )

        cursor = conn.cursor()
        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(20)

        results = [dict(zip(columns, row)) for row in rows]
        return results

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@tool
@traceable(name="query_athena_tool")
async def query_athena_tool(sql: str) -> str:
    """
    Executa consultas SQL no AWS Athena para análise de prontuários médicos.
    A query deve ser compatível com Presto/Athena.
    Retorne apenas dados relevantes. Limite sempre a 20 linhas.
    """
    # Validate SQL before execution
    try:
        validate_sql(sql)
    except ValueError as e:
        logger.warning(f"SQL inválido rejeitado: {e}")
        return f"Consulta inválida: {str(e)}"

    logger.info(f"Ferramenta Athena executando (async): {sql}")

    try:
        # Executa a query síncrona em uma thread separada para não bloquear o loop de eventos
        results = await asyncio.to_thread(_execute_athena_query, sql)

        # Captura os dados brutos no contexto para uso pelo Agente Avaliador
        captured = athena_results_context.get([])
        athena_results_context.set(captured + [{"sql": sql, "results": results}])

        if not results:
            logger.info("Ferramenta Athena: Nenhum resultado encontrado.")
            return "Nenhum resultado encontrado para esta consulta."

        logger.info(f"Ferramenta Athena: Retornadas {len(results)} linhas com sucesso.")
        return str(results)

    except Exception as e:
        logger.exception("Erro na ferramenta Athena")
        return f"Erro ao acessar o banco de dados Athena: {str(e)}. Verifique se as credenciais e o nome do banco estão corretos."