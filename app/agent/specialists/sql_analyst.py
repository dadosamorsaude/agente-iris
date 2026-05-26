"""
SQL Query Analyst & Executor Agent - Especialista Iris

Responsável por gerar, validar e executar consultas SQL no AWS Athena
para a análise de dados de cirurgia de catarata. Recebe o rag_context
clínico e retorna resultados estruturados com resumo quantitativo e
registros individuais conforme o output_mode solicitado.
"""

import json
import logging
import re
from datetime import datetime, timedelta, date
from app.tools.athena import _execute_athena_query, validate_sql, athena_results_context
from app.services.llm import get_chat_model_openai

logger = logging.getLogger(__name__)

# Schema oficial da tabela de catarata do N8N
CATARATA_SCHEMA = """
Tabela principal: pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
Dialeto: Athena/Presto SQL

Colunas disponíveis:
- id_paciente: bigint
- nome_paciente: string
- id_atendimento: bigint
- data_atendimento: date
- id_especialidade: bigint (Sempre filtrar por id_especialidade = 661)
- especialidade: string
- anamnese: string
- conduta: string
- hipotese_diagnostica: string
- observacao: string
- orientacao: string
- solicitacao: string
- especialidade_destino: string
- cid_codigo: string
- cid_descricao_detalhada: string
- id_clinica: bigint
- clinica: string
- regional: string
- uf: string
- municipio: string
- id_profissional: bigint
- nome_profissional: string
- prontuario_assinado: int
- id_exame_solicitado: bigint
- exame_solicitado: string
- prescricao: string
- posologia: string
- obs_atend_oftalmo: string
- flg_prescricao_cirurgica: string
- atestado: string

Campos narrativos/textuais para busca clínica (narrative_fields):
- anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao, exame_solicitado, prescricao, posologia, obs_atend_oftalmo, atestado, cid_descricao_detalhada

Filtros obrigatórios SEMPRE:
1. id_especialidade = 661

Regras SQL Athena/Presto:
- NUNCA use SELECT *
- Use lower(coalesce(campo, '')) para busca textual.
- Use regexp_like(lower(coalesce(campo, '')), 'padrao') para regex.
- Use regexp_extract(lower(coalesce(campo, '')), 'padrao', 1) para evidência.
- Use DATE 'YYYY-MM-DD' para datas literais.
- Não misture agregações e colunas detalhadas no mesmo SELECT sem separar em CTE.
- CTEs recomendadas para summary/rows: base -> texto_normalizado -> features -> score_calc -> classificado -> resumo -> SELECT final.
- CTEs recomendadas para sample: base -> texto_normalizado -> features -> score_calc -> classificado -> amostra (com row_number()) -> SELECT final WHERE rn <= sample_size.
"""


def _normalize_text_intent(text: str) -> str:
    if not text:
        return ""
    import unicodedata
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def _extract_period(original_input: str, hoje: str) -> dict:
    """
    Extrai o período temporal da pergunta conforme implementado no nó 'Preparar Contrato SQL' do n8n.
    """
    text = _normalize_text_intent(original_input)
    
    try:
        today = datetime.strptime(hoje, "%Y-%m-%d")
    except Exception:
        today = datetime.now()

    def iso_date(d: datetime) -> str:
        return d.strftime("%Y-%m-%d")

    # 1. Base inteira solicitada
    all_time_patterns = [
        "base inteira", "sem filtro de periodo", "sem periodo",
        "sem filtro temporal", "todo historico", "historico completo", "todos os dados"
    ]
    if any(p in text for p in all_time_patterns):
        return {"status": "all_time_requested", "start": None, "end_exclusive": None, "sql_filter": ""}

    # 2. Datas explícitas (YYYY-MM-DD ou DD/MM/YYYY)
    date_matches_iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", original_input)
    date_matches_br = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", original_input)
    
    explicit_dates = []
    for m in date_matches_iso:
        try:
            explicit_dates.append(datetime(int(m[0]), int(m[1]), int(m[2])))
        except:
            pass
    for m in date_matches_br:
        try:
            explicit_dates.append(datetime(int(m[2]), int(m[1]), int(m[0])))
        except:
            pass

    if len(explicit_dates) >= 2:
        start = explicit_dates[0]
        end = explicit_dates[1] + timedelta(days=1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }
    elif len(explicit_dates) == 1:
        start = explicit_dates[0]
        end = start + timedelta(days=1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }

    # 3. Meses por extenso
    months = {
        "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12
    }
    for month_name, month_index in months.items():
        if month_name in text:
            year_match = re.search(r"\b(20\d{2})\b", text)
            year = int(year_match.group(1)) if year_match else today.year
            start = datetime(year, month_index, 1)
            if month_index == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month_index + 1, 1)
            return {
                "status": "period_found",
                "start": iso_date(start),
                "end_exclusive": iso_date(end),
                "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
            }

    # 4. Últimos N dias
    last_days_match = re.search(r"\bultimos?\s+(\d+)\s+dias?\b", text)
    if last_days_match:
        days = max(1, int(last_days_match.group(1)))
        start = today - timedelta(days=days)
        end = today + timedelta(days=1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }

    # 5. Hoje/Ontem
    if "hoje" in text:
        start = today
        end = today + timedelta(days=1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }
    if "ontem" in text:
        start = today - timedelta(days=1)
        end = today
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }

    # 6. Mês atual / Mês passado
    if any(p in text for p in ["mes atual", "este mes", "neste mes"]):
        start = datetime(today.year, today.month, 1)
        if today.month == 12:
            end = datetime(today.year + 1, 1, 1)
        else:
            end = datetime(today.year, today.month + 1, 1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }
    if "mes passado" in text:
        if today.month == 1:
            start = datetime(today.year - 1, 12, 1)
            end = datetime(today.year, 1, 1)
        else:
            start = datetime(today.year, today.month - 1, 1)
            end = datetime(today.year, today.month, 1)
        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": f"data_atendimento >= DATE '{iso_date(start)}' AND data_atendimento < DATE '{iso_date(end)}'"
        }

    return {"status": "missing_period", "start": None, "end_exclusive": None, "sql_filter": None}


def _generate_sql(query: str, rag_context: str, output_mode: str, sample_size: int, hoje: str) -> str:
    """
    Usa o LLM para gerar uma query SQL válida para o Athena com base na pergunta do usuário,
    no contrato de dados do N8N e nas regras clínicas do RAG.
    """
    period = _extract_period(query, hoje)
    period_filter = f"\n3. Filtro temporal extraído: {period.get('sql_filter')}" if period.get("sql_filter") else ""
    
    llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")

    system_prompt = f"""Você é um gerador de SQL Athena/Presto especializado em auditoria clínica de cirurgias de catarata.
Gere UMA ÚNICA consulta SQL Athena limpa de acordo com o seguinte contrato:

{CATARATA_SCHEMA}

Contrato de Execução:
1. Tabela: pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
2. Filtro fixo obrigatório: id_especialidade = 661{period_filter}
4. Modo de saída (output_mode): '{output_mode}'
   - 'summary': Retorne apenas métricas agregadas agregadas no SELECT final:
     - total_registros (COUNT)
     - total_pacientes_unicos (COUNT DISTINCT id_paciente)
     - positivos (soma de classificação = 'positivo')
     - provaveis (soma de classificação = 'provável')
     - negativos (soma de classificação = 'negativo')
     - pos_operatorios (soma de classificação = 'pós-operatório')
     - percentual_positivos, percentual_provaveis, percentual_negativos
   - 'rows': Retorne as colunas agregadas acima E liste os detalhes das linhas individuais (com id_atendimento, id_paciente, data_atendimento, classificacao, score, campo_origem, termo_detectado, trecho_evidencia, cid_codigo_txt, flg_cirurgica, nome_profissional). Para retornar ambos juntos, use JOIN ou CROSS JOIN de forma limpa entre a CTE resumo e a CTE classificado. Limite as linhas de detalhes a 20.
   - 'sample': Ignore agregações (total_registros, percentual, etc). Retorne apenas linhas detalhadas individuais de amostra. Para limitar a amostra de forma correta e sem usar LIMIT, use row_number() em uma CTE 'amostra' e filtre rn <= {sample_size} no SELECT final.

5. Chave Clínica (Régua de classificação RAG):
{rag_context}

Regras Cruciais:
- NUNCA use SELECT *
- Use lower(coalesce(campo, '')) para busca textual.
- Nunca crie ou infira colunas que não existem no schema. Termos de RAG, léxicos, scores, etc., são construídos usando expressões lógicas (CASE WHEN) em CTEs e não são colunas físicas.
- Não use ILIKE, ::tipo, QUALIFY, DATEADD, GETDATE, REGEXP_CONTAINS, TOP, SAFE_CAST, ou regexp_instr.
- score e classificacao devem ser criados em CTEs progressivas. Nunca use um alias criado na mesma cláusula SELECT.
- A data de referência atual é: {hoje}.

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
    Formata e estrutura os resultados brutos do Athena em um payload padronizado,
    idêntico ao processamento do N8N.
    """
    if not results:
        return {
            "execution_status": "success",
            "summary": {
                "total_registros": 0,
                "total_pacientes_unicos": 0,
                "positivos": 0,
                "provaveis": 0,
                "negativos": 0,
                "pos_operatorios": 0,
                "percentual_positivos": 0.0,
                "percentual_provaveis": 0.0,
                "percentual_negativos": 0.0
            },
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": None,
            "limitations": ["Nenhum registro encontrado para os critérios informados."]
        }

    first = results[0]
    
    # Extrai o resumo analítico quantitativo
    summary = {
        "total_registros": first.get("total_registros") or first.get("total") or len(results),
        "total_pacientes_unicos": first.get("total_pacientes_unicos") or first.get("pacientes_unicos"),
        "positivos": first.get("positivos") or first.get("positivo"),
        "provaveis": first.get("provaveis") or first.get("provável"),
        "negativos": first.get("negativos") or first.get("negativo"),
        "pos_operatorios": first.get("pos_operatorios") or first.get("pós_operatorios"),
        "percentual_positivos": first.get("percentual_positivos") or first.get("pct_positivos"),
        "percentual_provaveis": first.get("percentual_provaveis") or first.get("pct_provaveis"),
        "percentual_negativos": first.get("percentual_negativos") or first.get("pct_negativos")
    }

    # Remove campos nulos/não preenchidos
    summary = {k: v for k, v in summary.items() if v is not None}

    if output_mode == "summary":
        return {
            "execution_status": "success",
            "summary": summary,
            "rows": [],
            "row_count": len(results),
            "truncated": False,
            "error": None,
            "limitations": []
        }

    # Para modos rows e sample, normaliza as linhas individuais de detalhes
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
        "summary": summary if output_mode == "rows" else {"total_registros": len(results)},
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

        # Retry: tenta uma única vez com o SQL corrigido pelo LLM (compatibilidade N8N)
        if retry_count < 1:
            logger.info("SQL Analyst: Tentando corrigir SQL e reexecutar (retry 1)...")
            try:
                llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")
                fix_prompt = (
                    f"O SQL abaixo causou este erro no AWS Athena (Presto):\n\n"
                    f"SQL:\n{sql}\n\nErro:\n{error_msg}\n\n"
                    f"Corrija o SQL para ser compatível com Presto/Athena e respeitar o schema de pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia. "
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

