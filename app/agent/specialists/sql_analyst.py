"""
SQL Query Analyst & Executor Agent - Especialista Iris

Responsável por gerar, validar e executar consultas SQL no AWS Athena
para a análise de dados de cirurgia de catarata. Recebe o rag_context
clínico e retorna resultados estruturados com resumo quantitativo e
registros individuais conforme o output_mode solicitado.
"""

import json
import logging
from app.tools.athena import _execute_athena_query, validate_sql, athena_results_context
from app.services.llm import get_chat_model_openai

logger = logging.getLogger(__name__)

# Schema da tabela de catarata
CATARATA_SCHEMA = """
Tabela principal: pdgt_amorsaude_inteligencia.tb_qualidade_prontuarios

Colunas disponíveis:
- id_agendamento, id_atendimento, data_atendimento
- status_agendamento, id_procedimento, id_especialidade, especialidade
- anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao
- especialidade_destino, cid_codigo, cid_codigo_txt (alias disponível), cid_descricao_detalhada
- id_clinica, clinica, regional, uf
- id_profissional, nome_profissional, prontuario_assinado

Filtros obrigatórios SEMPRE:
1. status_agendamento IN (4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 24, 40, 60, 83)
2. id_especialidade NOT IN (932, 1154, 993, 776, 777, 892, 1013, 711, 778, 658, 712, 732, 680, 1274, 779)

Regras SQL:
- NUNCA use SELECT *
- Use Presto/Athena SQL (use DATE '...' para datas, regexp_like() para regex, TRY(CAST()) para conversões seguras)
- Limite registros individuais com LIMIT (amostras: usar sample_size, demais: máx 20)
- Use funções de agregação (COUNT, SUM, AVG) para relatórios
- Para LIKE texto, use LOWER() para case-insensitive
- Para campos de texto como anamnese/conduta use regexp_like() ou LOWER() LIKE '%termo%'
"""


def _generate_sql(query: str, rag_context: str, output_mode: str, sample_size: int, hoje: str) -> str:
    """
    Usa o LLM para gerar uma query SQL válida para o Athena com base na pergunta do usuário
    e nas regras clínicas do RAG.
    """
    llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")

    system_prompt = f"""Você é um especialista em SQL para AWS Athena (Presto).
Gere UMA ÚNICA consulta SQL para responder à pergunta do usuário sobre cirurgia de catarata.

{CATARATA_SCHEMA}

Output mode: {output_mode}
- 'summary': Retorne apenas métricas agregadas (COUNT, SUM, percentuais). NÃO retorne linhas individuais.
- 'rows': Retorne detalhes de atendimentos individuais com campos clínicos relevantes. Limite 20 linhas.
- 'sample': Retorne {sample_size} registros individuais aleatórios com campos detalhados.

Data de referência: {hoje}

Regras de Cirurgia de Catarata (régua clínica do RAG):
{rag_context}

RETORNE APENAS O SQL PURO, sem markdown, sem explicações, sem comentários."""

    user_message = f"Pergunta: {query}\nGere a query SQL:"

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    sql = response.content.strip()
    # Remove blocos de código markdown se houver
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
        sql = sql.rsplit("```", 1)[0] if "```" in sql else sql
    return sql.strip()


def _format_sql_result(results: list[dict], output_mode: str, sample_size: int) -> dict:
    """
    Formata e estrutura os resultados brutos do Athena em um payload padronizado.
    """
    if not results:
        return {
            "execution_status": "success",
            "summary": {},
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": None,
            "limitations": ["Nenhum registro encontrado para os critérios informados."]
        }

    # Para modo summary, os resultados já são métricas agregadas
    if output_mode == "summary":
        summary = results[0] if len(results) == 1 else {}
        # Se tiver múltiplas linhas de agregação, consolida
        if len(results) > 1:
            summary = {
                "total_registros": len(results),
                "dados": results
            }

        return {
            "execution_status": "success",
            "summary": summary,
            "rows": [],
            "row_count": len(results),
            "truncated": False,
            "error": None,
            "limitations": []
        }

    # Para modos rows e sample, normaliza as linhas
    max_rows = sample_size if output_mode == "sample" else 20
    truncated = len(results) > max_rows
    rows_to_return = results[:max_rows]

    compact_rows = []
    for row in rows_to_return:
        compact_rows.append({
            "estrato": row.get("estrato") or row.get("classificacao") or row.get("classe") or None,
            "classificacao": row.get("classificacao") or row.get("estrato") or row.get("classe") or None,
            "id_atendimento": row.get("id_atendimento"),
            "id_paciente": row.get("id_paciente"),
            "data_atendimento": str(row.get("data_atendimento", "")) if row.get("data_atendimento") else None,
            "campo_origem": row.get("campo_origem") or row.get("campo") or None,
            "termo_detectado": row.get("termo_detectado") or row.get("termo") or None,
            "trecho_evidencia": str(row.get("trecho_evidencia") or row.get("evidencia") or "")[:700] or None,
            "score": row.get("score"),
            "cid_codigo_txt": row.get("cid_codigo_txt") or row.get("cid_codigo") or None,
            "flg_cirurgica": row.get("flg_cirurgica"),
            "nome_profissional": row.get("nome_profissional"),
        })

    return {
        "execution_status": "success",
        "summary": {"total_registros": len(results)},
        "rows": compact_rows,
        "row_count": len(results),
        "truncated": truncated,
        "error": None,
        "limitations": ["Resultados truncados." ] if truncated else []
    }


def sql_analyst_expert(
    query: str,
    rag_context: str,
    output_mode: str = "summary",
    sample_size: int = 5,
    hoje: str = "",
    retry_count: int = 0
) -> dict:
    """
    Especialista SQL para o sistema Iris.
    Gera, valida e executa uma query Athena, retornando o payload estruturado.
    Se a execução falhar, tenta gerar e reexecutar uma nova query corrigida (1 retry).
    """
    logger.info(f"SQL Analyst Expert iniciado | output_mode={output_mode} | sample_size={sample_size}")

    try:
        sql = _generate_sql(query, rag_context, output_mode, sample_size, hoje)
        logger.info(f"SQL gerado:\n{sql}")
    except Exception as e:
        logger.error(f"Falha ao gerar SQL: {e}")
        return {
            "execution_status": "error",
            "error": {"type": "sql_generation_error", "message": str(e)},
            "summary": {},
            "rows": [],
            "row_count": 0,
            "sql": None,
            "limitations": ["Falha ao gerar o SQL."]
        }

    try:
        validate_sql(sql)
    except ValueError as e:
        logger.warning(f"SQL inválido: {e}")
        return {
            "execution_status": "error",
            "error": {"type": "sql_validation_error", "message": str(e)},
            "summary": {},
            "rows": [],
            "row_count": 0,
            "sql": sql,
            "limitations": [str(e)]
        }

    try:
        results = _execute_athena_query(sql)

        # Captura os dados brutos no contexto para uso pelo Agente Avaliador
        captured = athena_results_context.get([])
        athena_results_context.set(captured + [{"sql": sql, "results": results}])

        payload = _format_sql_result(results, output_mode, sample_size)
        payload["sql"] = sql
        logger.info(f"SQL Analyst: Execução bem-sucedida | row_count={payload['row_count']}")
        return payload

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Erro ao executar SQL no Athena: {error_msg}")

        # Retry: tenta uma única vez com o SQL corrigido pelo LLM
        if retry_count < 1:
            logger.info("SQL Analyst: Tentando corrigir SQL e reexecutar (retry 1)...")
            try:
                llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")
                fix_prompt = (
                    f"O SQL abaixo causou este erro no AWS Athena (Presto):\n\n"
                    f"SQL:\n{sql}\n\nErro:\n{error_msg}\n\n"
                    f"Corrija o SQL para ser compatível com Presto/Athena. "
                    f"Retorne APENAS o SQL corrigido, sem markdown e sem explicações."
                )
                fix_response = llm.invoke([{"role": "user", "content": fix_prompt}])
                fixed_sql = fix_response.content.strip()
                if fixed_sql.startswith("```"):
                    fixed_sql = fixed_sql.split("```")[1]
                    if fixed_sql.lower().startswith("sql"):
                        fixed_sql = fixed_sql[3:]
                    fixed_sql = fixed_sql.rsplit("```", 1)[0] if "```" in fixed_sql else fixed_sql
                fixed_sql = fixed_sql.strip()

                validate_sql(fixed_sql)
                results = _execute_athena_query(fixed_sql)
                captured = athena_results_context.get([])
                athena_results_context.set(captured + [{"sql": fixed_sql, "results": results}])
                payload = _format_sql_result(results, output_mode, sample_size)
                payload["sql"] = fixed_sql
                payload["limitations"] = payload.get("limitations", []) + ["SQL foi auto-corrigido na execução."]
                logger.info(f"SQL Analyst: Retry bem-sucedido | row_count={payload['row_count']}")
                return payload
            except Exception as retry_err:
                logger.error(f"SQL Analyst: Retry também falhou: {retry_err}")
                return {
                    "execution_status": "error",
                    "error": {"type": "sql_executor_error", "message": f"Original: {error_msg}. Retry: {retry_err}"},
                    "summary": {},
                    "rows": [],
                    "row_count": 0,
                    "sql": sql,
                    "limitations": ["Execução SQL falhou mesmo após tentativa de autocorreção."]
                }

        return {
            "execution_status": "error",
            "error": {"type": "sql_executor_error", "message": error_msg},
            "summary": {},
            "rows": [],
            "row_count": 0,
            "sql": sql,
            "limitations": ["Execução SQL falhou."]
        }
