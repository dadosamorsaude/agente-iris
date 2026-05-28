"""
SQL Query Analyst & Executor Agent - Especialista Iris

Responsável por gerar, validar e executar consultas SQL no AWS Athena
para a análise de dados de cirurgia de catarata.

Melhorias principais:
- Detecta automaticamente a intenção da pergunta do usuário.
- Só usa agregações quando o usuário realmente pede total, percentual,
  distribuição, ranking, agrupamento ou evolução temporal.
- Retorna linhas detalhadas quando o usuário pede listagem, casos, pacientes,
  atendimentos, exemplos ou amostras.
- Adiciona validação semântica para impedir agregações indevidas.
- Preserva os dados brutos retornados pelo Athena em modos detalhados.
"""

import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, date
from typing import Any, Optional

from app.core.observability import get_langsmith_callbacks, traceable
from app.services.intent import normalize_text
from app.tools.athena import query_athena_tool, validate_sql, athena_results_context
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
- anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao,
  exame_solicitado, prescricao, posologia, obs_atend_oftalmo, atestado,
  cid_descricao_detalhada

Filtros obrigatórios SEMPRE:
1. id_especialidade = 661

Regras SQL Athena/Presto:
- NUNCA use SELECT *
- Use lower(coalesce(campo, '')) para busca textual.
- Use regexp_like(lower(coalesce(campo, '')), 'padrao') para regex.
- Use regexp_extract(lower(coalesce(campo, '')), 'padrao', 1) para evidência.
- Use DATE 'YYYY-MM-DD' para datas literais.
- Não misture agregações e colunas detalhadas no mesmo SELECT sem separar em CTE.
- CTEs recomendadas para classificação clínica:
  base -> texto_normalizado -> features -> score_calc -> classificado.
- Quando precisar limitar linhas detalhadas, prefira row_number() em CTE
  e filtre rn <= N no SELECT final.
"""


# normalize_text importado de app.services.intent


def _strip_markdown_sql(sql: str) -> str:
    if not sql:
        return ""

    sql = sql.strip()

    if sql.startswith("```"):
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1].strip()

        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

    return sql.strip()


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]

    return value


def _detect_requested_limit(text: str, default_limit: int) -> int:
    normalized = normalize_text(text)

    patterns = [
        r"\b(?:top|primeiros?|primeiras?|ultimos?|ultimas?)\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s+(?:linhas?|registros?|casos?|exemplos?|amostras?|pacientes?|atendimentos?)\b",
        r"\b(?:traga|mostre|liste|retorne)\s+(\d{1,3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                value = int(match.group(1))
                return max(1, min(value, 200))
            except Exception:
                pass

    return default_limit


def _extract_period(original_input: str, hoje: str) -> dict:
    """
    Extrai o período temporal da pergunta conforme implementado no nó
    'Preparar Contrato SQL' do n8n.
    """
    text = normalize_text(original_input)

    try:
        today = datetime.strptime(hoje, "%Y-%m-%d")
    except Exception:
        today = datetime.now()

    def iso_date(d: datetime) -> str:
        return d.strftime("%Y-%m-%d")

    # 1. Base inteira solicitada
    all_time_patterns = [
        "base inteira",
        "sem filtro de periodo",
        "sem periodo",
        "sem filtro temporal",
        "todo historico",
        "historico completo",
        "todos os dados",
    ]

    if any(p in text for p in all_time_patterns):
        return {
            "status": "all_time_requested",
            "start": None,
            "end_exclusive": None,
            "sql_filter": "",
        }

    # 2. Datas explícitas: YYYY-MM-DD ou DD/MM/YYYY
    date_matches_iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", original_input)
    date_matches_br = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", original_input)

    explicit_dates = []

    for m in date_matches_iso:
        try:
            explicit_dates.append(datetime(int(m[0]), int(m[1]), int(m[2])))
        except Exception:
            pass

    for m in date_matches_br:
        try:
            explicit_dates.append(datetime(int(m[2]), int(m[1]), int(m[0])))
        except Exception:
            pass

    if len(explicit_dates) >= 2:
        start = explicit_dates[0]
        end = explicit_dates[1] + timedelta(days=1)

        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
        }

    if len(explicit_dates) == 1:
        start = explicit_dates[0]
        end = start + timedelta(days=1)

        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
        }

    # 3. Meses por extenso
    months = {
        "janeiro": 1,
        "fevereiro": 2,
        "marco": 3,
        "março": 3,
        "abril": 4,
        "maio": 5,
        "junho": 6,
        "julho": 7,
        "agosto": 8,
        "setembro": 9,
        "outubro": 10,
        "novembro": 11,
        "dezembro": 12,
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
                "sql_filter": (
                    f"data_atendimento >= DATE '{iso_date(start)}' "
                    f"AND data_atendimento < DATE '{iso_date(end)}'"
                ),
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
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
        }

    # 5. Hoje/Ontem
    if "hoje" in text:
        start = today
        end = today + timedelta(days=1)

        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
        }

    if "ontem" in text:
        start = today - timedelta(days=1)
        end = today

        return {
            "status": "period_found",
            "start": iso_date(start),
            "end_exclusive": iso_date(end),
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
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
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
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
            "sql_filter": (
                f"data_atendimento >= DATE '{iso_date(start)}' "
                f"AND data_atendimento < DATE '{iso_date(end)}'"
            ),
        }

    return {
        "status": "missing_period",
        "start": None,
        "end_exclusive": None,
        "sql_filter": None,
    }


def _resolve_query_shape(
    query: str,
    output_mode: Optional[str] = None,
    sample_size: int = 5,
) -> dict:
    """Resolve apenas a preferencia inicial da consulta.

    O SQL fica livre para adaptar a consulta a pergunta do usuario. Esta funcao
    evita o antigo roteamento grande e deixa so tres modos: detail, aggregate
    e mixed. Na duvida, detail preserva dados brutos.
    """
    text = normalize_text(query)
    explicit_mode = normalize_text(output_mode or "")

    if explicit_mode in {"detail", "detalhe", "detalhado", "sample", "amostra", "lookup"}:
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Modo detalhado solicitado explicitamente.",
        }

    if explicit_mode in {"summary", "aggregate", "agregado", "resumo"}:
        return {
            "query_shape": "aggregate",
            "output_mode": "summary",
            "limit": 0,
            "reason": "Modo agregado solicitado explicitamente.",
        }

    if explicit_mode in {"mixed", "resumo_linhas", "summary_rows"}:
        return {
            "query_shape": "mixed",
            "output_mode": "mixed",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Modo misto solicitado explicitamente.",
        }

    wants_detail = re.search(
        r"\b(liste|listar|lista|mostre|mostrar|traga|retorne|casos|registros|linhas|"
        r"pacientes|atendimentos|evidencias|detalhes|prontuarios|amostra|exemplos?)\b",
        text,
    )
    wants_aggregate = re.search(
        r"\b(quantos|quantas|total|percentual|porcentagem|proporcao|taxa|media|"
        r"contagem|distribuicao|volume|indicador|ranking|evolucao|por clinica|"
        r"por regional|por uf|por municipio|por profissional|por medico|por mes|mensal)\b",
        text,
    )

    if wants_detail and wants_aggregate:
        return {
            "query_shape": "mixed",
            "output_mode": "mixed",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Pedido combina metrica e dados brutos.",
        }

    if wants_detail or re.search(r"\b\d{5,}\b", text):
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Pedido pede ou sugere registros individuais.",
        }

    if wants_aggregate:
        return {
            "query_shape": "aggregate",
            "output_mode": "summary",
            "limit": 0,
            "reason": "Pedido pede metrica, agrupamento ou evolucao.",
        }

    return {
        "query_shape": "detail",
        "output_mode": "detail",
        "limit": _detect_requested_limit(text, 20),
        "reason": "Fallback simples: preservar dados brutos.",
    }


def _build_query_shape_contract(query_shape: str, output_mode: str, sample_size: int, detail_limit: int) -> str:
    return f"""
Preferencia inicial de consulta: {query_shape}
Modo de saida preferido: {output_mode}
Limite preferido para linhas detalhadas: {detail_limit}

Use a pergunta do usuario como fonte principal de decisao. Voce pode retornar:
- aggregate: metricas, totais, percentuais, rankings, recortes ou series temporais.
- detail: registros brutos/individuais, amostras integrais, pacientes, atendimentos, evidencias.
- mixed: metricas e registros brutos juntos quando isso responder melhor ao pedido.

Regras de liberdade controlada:
- Se o usuario pedir dados brutos, registros, casos, pacientes, atendimentos, amostra ou evidencias, preserve linhas individuais.
- Para registros individuais, inclua SEMPRE os campos textuais/narrativos: anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao, prescricao, exame_solicitado, obs_atend_oftalmo, cid_descricao_detalhada, alem de ids, data, paciente, profissional e clinica.
- Para metricas, use agregacoes livremente, mas mantenha nomes de colunas claros.
- Para respostas mistas, separe resumo e detalhes em CTEs e retorne ambos no resultado final.
- Se precisar limitar registros brutos, prefira row_number() em CTE e filtre rn <= {detail_limit}.
- Nao use LIMIT.
"""


# Colunas padrão para substituir SELECT * automaticamente
_DEFAULT_COLUMNS = [
    "id_atendimento", "id_paciente", "nome_paciente", "data_atendimento",
    "id_especialidade", "especialidade",
    "anamnese", "conduta", "hipotese_diagnostica",
    "cid_codigo", "cid_descricao_detalhada",
    "id_clinica", "clinica", "regional", "uf", "municipio",
    "id_profissional", "nome_profissional",
    "prontuario_assinado",
    "prescricao", "flg_prescricao_cirurgica",
    "solicitacao", "exame_solicitado",
    "observacao", "orientacao", "obs_atend_oftalmo",
    "atestado", "posologia",
]


def _replace_select_star(sql: str) -> str:
    """Substitui SELECT * por colunas explícitas."""
    cols = ",\n        ".join(_DEFAULT_COLUMNS)
    return re.sub(
        r'\bselect\s+\*\b',
        f"SELECT\n        {cols}",
        sql,
        flags=re.IGNORECASE,
    )


def _semantic_validate_sql(sql: str, query_shape: str, original_query: str) -> str:
    """Valida e corrige o SQL. Retorna o SQL corrigido (pode substituir SELECT *)."""
    sql_normalized = normalize_text(sql)

    if not sql_normalized:
        raise ValueError("SQL vazio gerado pelo LLM.")

    if re.search(r"\bselect\s+\*", sql_normalized):
        corrected = _replace_select_star(sql)
        logger.info("SELECT * substituído por colunas explícitas")
        return corrected

    if "id_especialidade = 661" not in sql_normalized and "id_especialidade=661" not in sql_normalized:
        raise ValueError("SQL invalido semanticamente: filtro obrigatorio id_especialidade = 661 ausente.")

    return sql


@traceable(name="generate_sql", as_type="llm")
async def _generate_sql(
    query: str,
    rag_context: str,
    output_mode: str,
    sample_size: int,
    hoje: str,
    query_shape: str = "detail",
    detail_limit: int = 20,
    extra_instruction: str = "",
) -> str:
    """Usa o LLM para gerar uma query SQL válida para o Athena."""
    period = _extract_period(query, hoje)

    period_filter = ""
    if period.get("sql_filter"):
        period_filter = f"\n3. Filtro temporal extraído: {period.get('sql_filter')}"

    query_shape_contract = _build_query_shape_contract(
        query_shape=query_shape,
        output_mode=output_mode,
        sample_size=sample_size,
        detail_limit=detail_limit,
    )

    llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")

    system_prompt = f"""Você é um gerador de SQL Athena/Presto especializado em auditoria clínica de cirurgias de catarata.

Gere UMA ÚNICA consulta SQL Athena limpa, adaptada à intenção real da pergunta do usuário.

REGRA MAIS IMPORTANTE:
- Decida livremente a melhor consulta para responder ao usuario.
- Use agregacoes quando elas ajudarem.
- Preserve registros brutos quando o usuario pedir casos, amostras, pacientes, atendimentos, prontuarios ou evidencias.
- Consultas mistas sao permitidas: resumo + linhas individuais.

{CATARATA_SCHEMA}

Contrato de Execução:
1. Tabela: pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
2. Filtro fixo obrigatório: id_especialidade = 661{period_filter}
4. output_mode preferido: '{output_mode}'
5. query_shape preferido: '{query_shape}'

{query_shape_contract}

Chave Clínica / Régua de classificação RAG:
{rag_context}

Regras Cruciais:
- NUNCA use SELECT *
- Nunca crie ou infira colunas que não existem no schema.
- Termos de RAG, léxicos, scores e classificações devem ser construídos
  usando expressões lógicas em CTEs, não como colunas físicas.
- Não use ILIKE, ::tipo, QUALIFY, DATEADD, GETDATE, REGEXP_CONTAINS, TOP,
  SAFE_CAST ou regexp_instr.
- score e classificacao devem ser criados em CTEs progressivas.
- Nunca use um alias criado na mesma cláusula SELECT.
- Para texto, use lower(coalesce(campo, '')).
- Para datas literais, use DATE 'YYYY-MM-DD'.
- A data de referência atual é: {hoje}.
- Se nenhum período foi solicitado, não invente filtro temporal.
- Retorne APENAS o SQL puro, sem markdown, sem explicações, sem comentários.

{extra_instruction}
"""

    user_message = f"Pergunta: {query}\nGere a query SQL:"

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        config={"callbacks": get_langsmith_callbacks()},
    )

    return _strip_markdown_sql(response.content)


@traceable(name="repair_sql_semantics", as_type="llm")
async def _repair_sql_for_semantics(
    original_sql: str,
    semantic_error: str,
    query: str,
    rag_context: str,
    output_mode: str,
    sample_size: int,
    hoje: str,
    query_shape: str,
    detail_limit: int,
) -> str:
    extra_instruction = f"""
O SQL anterior foi rejeitado pela validação semântica:

Erro:
{semantic_error}

SQL anterior:
{original_sql}

Regenere a consulta do zero. NUNCA use SELECT * — liste as colunas explícitas do schema relevantes para a pergunta. O query_shape '{query_shape}' e apenas uma preferencia; responda a pergunta do usuario com a forma de dados mais util.
"""

    return await _generate_sql(
        query=query,
        rag_context=rag_context,
        output_mode=output_mode,
        sample_size=sample_size,
        hoje=hoje,
        query_shape=query_shape,
        detail_limit=detail_limit,
        extra_instruction=extra_instruction,
    )


def _extract_summary(first: dict, fallback_total: int) -> dict:
    summary = {
        "total_registros": first.get("total_registros") or first.get("total") or fallback_total,
        "total_pacientes_unicos": first.get("total_pacientes_unicos") or first.get("pacientes_unicos"),
        "positivos": first.get("positivos") or first.get("positivo"),
        "provaveis": first.get("provaveis") or first.get("provável"),
        "negativos": first.get("negativos") or first.get("negativo"),
        "pos_operatorios": first.get("pos_operatorios") or first.get("pós_operatorios"),
        "percentual_positivos": first.get("percentual_positivos") or first.get("pct_positivos"),
        "percentual_provaveis": first.get("percentual_provaveis") or first.get("pct_provaveis"),
        "percentual_negativos": first.get("percentual_negativos") or first.get("pct_negativos"),
    }

    return {k: _make_json_safe(v) for k, v in summary.items() if v is not None}


@traceable(name="repair_sql_execution", as_type="llm")
async def _repair_sql_for_execution(
    original_sql: str,
    error_msg: str,
    query: str,
    query_shape: str,
    output_mode: str,
    sample_size: int,
    detail_limit: int,
) -> str:
    """Tenta corrigir SQL que falhou na execução do Athena."""
    llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")

    shape_contract = _build_query_shape_contract(
        query_shape=query_shape,
        output_mode=output_mode,
        sample_size=sample_size,
        detail_limit=detail_limit,
    )

    fix_prompt = f"""O SQL abaixo causou erro no AWS Athena/Presto.

Pergunta original:
{query}

query_shape:
{query_shape}

output_mode:
{output_mode}

Contrato do tipo de consulta:
{shape_contract}

SQL com erro:
{original_sql}

Erro:
{error_msg}

Corrija o SQL para ser compatível com Presto/Athena e respeitar o schema:
pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia

Regras obrigatorias:
- Nunca usar SELECT *.
- Preserve dados brutos quando a pergunta pedir registros individuais.
- Retorne APENAS o SQL corrigido, sem markdown e sem explicacoes.
"""

    fix_response = await llm.ainvoke(
        [{"role": "user", "content": fix_prompt}],
        config={"callbacks": get_langsmith_callbacks()},
    )
    return _strip_markdown_sql(fix_response.content)


def _compact_detail_row(row: dict) -> dict:
    safe_row = _make_json_safe(row)

    return {
        "estrato": row.get("estrato") or row.get("classificacao") or row.get("classe"),
        "classificacao": row.get("classificacao") or row.get("estrato") or row.get("classe"),
        "id_atendimento": row.get("id_atendimento"),
        "id_paciente": row.get("id_paciente"),
        "nome_paciente": row.get("nome_paciente"),
        "data_atendimento": _make_json_safe(row.get("data_atendimento")),
        "campo_origem": row.get("campo_origem") or row.get("campo"),
        "termo_detectado": row.get("termo_detectado") or row.get("termo"),
        "trecho_evidencia": (
            str(row.get("trecho_evidencia") or row.get("evidencia") or "")[:700] or None
        ),
        "score": row.get("score"),
        "cid_codigo_txt": row.get("cid_codigo_txt") or row.get("cid_codigo"),
        "flg_cirurgica": row.get("flg_cirurgica"),
        "clinica": row.get("clinica"),
        "regional": row.get("regional"),
        "uf": row.get("uf"),
        "municipio": row.get("municipio"),
        "nome_profissional": row.get("nome_profissional"),
        "raw": safe_row,
    }


def _format_sql_result(
    results: list[dict],
    output_mode: str,
    sample_size: int,
    query_shape: str = "detail",
    detail_limit: int = 20,
) -> dict:
    """Formata o retorno preservando dados brutos sempre que existirem."""
    if not results:
        return {
            "execution_status": "success",
            "summary": {},
            "rows": [],
            "raw_rows": [],
            "row_count": 0,
            "total_rows_returned_by_athena": 0,
            "truncated": False,
            "error": None,
            "limitations": ["Nenhum registro encontrado para os criterios informados."],
        }

    max_rows = detail_limit if detail_limit > 0 else 200
    if query_shape == "aggregate" and output_mode == "summary":
        max_rows = max(len(results), 200)
    elif query_shape == "detail":
        max_rows = detail_limit
    elif query_shape == "mixed":
        max_rows = detail_limit

    rows_to_return = results[:max_rows]
    truncated = len(results) > len(rows_to_return)
    safe_rows = [_make_json_safe(row) for row in rows_to_return]

    payload = {
        "execution_status": "success",
        "summary": {},
        "rows": safe_rows,
        "raw_rows": safe_rows,
        "row_count": len(rows_to_return),
        "total_rows_returned_by_athena": len(results),
        "truncated": truncated,
        "error": None,
        "limitations": ["Resultados truncados."] if truncated else [],
    }

    if query_shape in {"aggregate", "mixed"} or output_mode in {"summary", "mixed"}:
        payload["summary"] = _extract_summary(results[0], len(results))

    return payload


async def _execute_query(sql: str) -> list[dict[str, Any]]:
    """Execute SQL via query_athena_tool com tracing."""
    result_str = await query_athena_tool.ainvoke({"sql": sql})

    if result_str.startswith("Consulta inválida") or result_str.startswith("Erro ao acessar"):
        raise ValueError(result_str)

    if result_str in ("Nenhum resultado encontrado para esta consulta.", ""):
        return []

    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Falha ao parsear resultado da tool: {e}")
        raise ValueError(f"Resultado inválido da ferramenta Athena: {result_str[:200]}")


@traceable(name="sql_analyst_expert", as_type="chain")
async def sql_analyst_expert(
    query: str,
    rag_context: str,
    output_mode: Optional[str] = None,
    sample_size: int = 5,
    hoje: str = "",
) -> dict:
    """
    Especialista SQL para o sistema Iris.

    Gera, valida e executa uma query Athena, retornando payload estruturado.
    Em caso de falha de execução, tenta corrigir o SQL automaticamente
    (até 3 tentativas no total).
    """
    hoje = hoje or date.today().strftime("%Y-%m-%d")

    intent = _resolve_query_shape(
        query=query,
        output_mode=output_mode,
        sample_size=sample_size,
    )

    query_shape = intent["query_shape"]
    resolved_output_mode = intent["output_mode"]
    detail_limit = intent["limit"] or 20

    logger.info(
        "SQL Analyst Expert iniciado | "
        f"query_shape={query_shape} | "
        f"output_mode={resolved_output_mode} | "
        f"sample_size={sample_size} | "
        f"detail_limit={detail_limit} | "
        f"reason={intent.get('reason')}"
    )

    try:
        sql = await _generate_sql(
            query=query,
            rag_context=rag_context,
            output_mode=resolved_output_mode,
            sample_size=sample_size,
            hoje=hoje,
            query_shape=query_shape,
            detail_limit=detail_limit,
        )

        logger.info(f"SQL gerado:\n{sql}")

        try:
            sql = _semantic_validate_sql(sql, query_shape, query)
        except ValueError as semantic_error:
            logger.warning(f"SQL rejeitado por validação semântica: {semantic_error}")

            sql = await _repair_sql_for_semantics(
                original_sql=sql,
                semantic_error=str(semantic_error),
                query=query,
                rag_context=rag_context,
                output_mode=resolved_output_mode,
                sample_size=sample_size,
                hoje=hoje,
                query_shape=query_shape,
                detail_limit=detail_limit,
            )

            logger.info(f"SQL regenerado após validação semântica:\n{sql}")

            try:
                sql = _semantic_validate_sql(sql, query_shape, query)
            except ValueError as e:
                logger.error(f"Repair também falhou validação semântica: {e}")

                return {
                    "execution_status": "error",
                    "error": {
                        "type": "sql_validation_error",
                        "message": f"Repair falhou: {e}",
                    },
                    "summary": {},
                    "rows": [],
                    "row_count": 0,
                    "sql": sql,
                    "query_shape": query_shape,
                    "output_mode": resolved_output_mode,
                    "intent_reason": intent.get("reason"),
                    "limitations": [f"Repair semântico falhou: {e}"],
                }

    except Exception as e:
        logger.error(f"Falha ao gerar SQL: {e}")

        return {
            "execution_status": "error",
            "error": {"type": "sql_generation_error", "message": str(e)},
            "summary": {},
            "rows": [],
            "row_count": 0,
            "sql": None,
            "query_shape": query_shape,
            "output_mode": resolved_output_mode,
            "intent_reason": intent.get("reason"),
            "limitations": ["Falha ao gerar o SQL."],
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
            "query_shape": query_shape,
            "output_mode": resolved_output_mode,
            "intent_reason": intent.get("reason"),
            "limitations": [str(e)],
        }

    current_sql = sql
    last_error = ""
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            results = await _execute_query(current_sql)

            captured = athena_results_context.get([])
            athena_results_context.set(captured + [{"sql": current_sql, "results": results}])

            payload = _format_sql_result(
                results=results,
                output_mode=resolved_output_mode,
                sample_size=sample_size,
                query_shape=query_shape,
                detail_limit=detail_limit,
            )

            payload["sql"] = current_sql
            payload["query_shape"] = query_shape
            payload["output_mode"] = resolved_output_mode
            payload["intent_reason"] = intent.get("reason")

            if attempt > 0:
                payload["limitations"] = payload.get("limitations", []) + [
                    f"SQL foi auto-corrigido na tentativa {attempt + 1}."
                ]

            logger.info(
                "SQL Analyst: Execução bem-sucedida | "
                f"query_shape={query_shape} | "
                f"row_count={payload.get('row_count')} | "
                f"tentativa={attempt + 1}"
            )

            return payload

        except Exception as e:
            last_error = str(e)
            logger.error(
                f"SQL Analyst: Tentativa {attempt + 1}/{max_attempts} falhou: {last_error}"
            )

            if attempt < max_attempts - 1:
                logger.info(f"SQL Analyst: Corrigindo SQL e reexecutando (tentativa {attempt + 2})...")
                current_sql = await _repair_sql_for_execution(
                    original_sql=current_sql,
                    error_msg=last_error,
                    query=query,
                    query_shape=query_shape,
                    output_mode=resolved_output_mode,
                    sample_size=sample_size,
                    detail_limit=detail_limit,
                )
                current_sql = _semantic_validate_sql(current_sql, query_shape, query)
                validate_sql(current_sql)

    logger.error(f"SQL Analyst: Todas as {max_attempts} tentativas falharam")

    return {
        "execution_status": "error",
        "error": {
            "type": "sql_executor_error",
            "message": f"Todas as {max_attempts} tentativas falharam. Último erro: {last_error}",
        },
        "summary": {},
        "rows": [],
        "row_count": 0,
        "sql": sql,
        "query_shape": query_shape,
        "output_mode": resolved_output_mode,
        "intent_reason": intent.get("reason"),
        "limitations": [
            "Execução SQL falhou mesmo após tentativa de autocorreção."
        ],
    }
