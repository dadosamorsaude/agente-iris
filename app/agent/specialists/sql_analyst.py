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

import logging
import re
import unicodedata
from datetime import datetime, timedelta, date
from typing import Any, Optional

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


AGGREGATION_REGEXES = [
    r"\bcount\s*\(",
    r"\bsum\s*\(",
    r"\bavg\s*\(",
    r"\bmin\s*\(",
    r"\bmax\s*\(",
    r"\bgroup\s+by\b",
    r"\bhaving\b",
]


DETAIL_QUERY_SHAPES = {"detail", "sample", "lookup"}
AGGREGATE_QUERY_SHAPES = {"aggregate", "breakdown", "timeline", "mixed"}


def _normalize_text_intent(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


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
    normalized = _normalize_text_intent(text)

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
    text = _normalize_text_intent(original_input)

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
    """
    Decide automaticamente o tipo de consulta esperado.

    query_shape:
    - aggregate: total, quantidade, percentual, soma geral
    - detail: linhas individuais, pacientes, atendimentos, casos
    - sample: amostra integral, exemplos
    - lookup: busca específica por paciente, atendimento, profissional ou valor
    - breakdown: agrupamento por clínica, regional, médico, UF etc.
    - timeline: evolução no tempo, mês a mês, dia a dia
    - mixed: resumo + detalhes
    - distinct: lista de valores distintos
    """
    text = _normalize_text_intent(query)
    explicit_mode = _normalize_text_intent(output_mode or "")
    requested_limit = _detect_requested_limit(text, sample_size)

    aggregate_terms = [
        "quantos",
        "quantas",
        "total",
        "totais",
        "percentual",
        "porcentagem",
        "proporcao",
        "proporção",
        "taxa",
        "media",
        "média",
        "mediana",
        "contagem",
        "distribuicao",
        "distribuição",
        "volume",
        "indicador",
        "indicadores",
    ]

    detail_terms = [
        "liste",
        "listar",
        "lista",
        "mostre",
        "mostrar",
        "traga",
        "retorne",
        "quais pacientes",
        "quais atendimentos",
        "casos",
        "registros",
        "linhas",
        "pacientes com",
        "atendimentos com",
        "evidencias",
        "evidências",
        "detalhe",
        "detalhes",
        "prontuarios",
        "prontuários",
    ]

    sample_terms = [
        "amostra",
        "amostras",
        "exemplo",
        "exemplos",
        "alguns casos",
        "algumas linhas",
        "linhas de exemplo",
        "amostra integral",
    ]

    timeline_terms = [
        "mes a mes",
        "mês a mês",
        "por mes",
        "por mês",
        "mensal",
        "dia a dia",
        "por dia",
        "diario",
        "diário",
        "semanal",
        "por semana",
        "evolucao",
        "evolução",
        "ao longo do tempo",
        "serie temporal",
        "série temporal",
        "tendencia",
        "tendência",
    ]

    breakdown_terms = [
        "por clinica",
        "por clínica",
        "por regional",
        "por uf",
        "por municipio",
        "por município",
        "por profissional",
        "por medico",
        "por médico",
        "por especialidade",
        "agrupado",
        "agrupada",
        "agrupamento",
        "ranking",
        "rank",
        "top",
    ]

    lookup_terms = [
        "id_paciente",
        "id atendimento",
        "id_atendimento",
        "id profissional",
        "id_profissional",
        "paciente especifico",
        "paciente específico",
        "atendimento especifico",
        "atendimento específico",
    ]

    distinct_terms = [
        "quais valores",
        "valores distintos",
        "valores unicos",
        "valores únicos",
        "distinct",
        "sem repetir",
        "unicos",
        "únicos",
        "quais clinicas",
        "quais clínicas",
        "quais regionais",
        "quais profissionais",
        "quais municipios",
        "quais municípios",
    ]

    has_aggregate = any(term in text for term in aggregate_terms)
    has_detail = any(term in text for term in detail_terms)
    has_sample = any(term in text for term in sample_terms)
    has_timeline = any(term in text for term in timeline_terms)
    has_breakdown = any(term in text for term in breakdown_terms)
    has_lookup = any(term in text for term in lookup_terms)
    has_distinct = any(term in text for term in distinct_terms)

    asks_summary_and_details = any(
        term in text
        for term in [
            "resumo e detalhe",
            "resumo e detalhes",
            "resumo com detalhes",
            "resumo com exemplos",
            "total e exemplos",
            "quantidade e exemplos",
            "contagem e exemplos",
            "com exemplos",
        ]
    )

    if explicit_mode in {"sample", "amostra"}:
        return {
            "query_shape": "sample",
            "output_mode": "sample",
            "limit": requested_limit,
            "reason": "output_mode explícito como sample.",
        }

    if explicit_mode in {"detail", "detalhe", "detalhado"}:
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 20),
            "reason": "output_mode explícito como detail.",
        }

    if explicit_mode in {"summary", "aggregate", "agregado", "resumo"}:
        return {
            "query_shape": "aggregate",
            "output_mode": "summary",
            "limit": 0,
            "reason": "output_mode explícito como summary.",
        }

    if explicit_mode in {"mixed", "resumo_linhas", "summary_rows"}:
        return {
            "query_shape": "mixed",
            "output_mode": "mixed",
            "limit": _detect_requested_limit(text, 20),
            "reason": "output_mode explícito como mixed.",
        }

    if has_sample:
        return {
            "query_shape": "sample",
            "output_mode": "sample",
            "limit": requested_limit,
            "reason": "Usuário pediu amostra/exemplos.",
        }

    if has_timeline:
        return {
            "query_shape": "timeline",
            "output_mode": "rows",
            "limit": 200,
            "reason": "Usuário pediu evolução temporal.",
        }

    if has_breakdown:
        return {
            "query_shape": "breakdown",
            "output_mode": "rows",
            "limit": 200,
            "reason": "Usuário pediu agrupamento/ranking.",
        }

    if asks_summary_and_details or (has_aggregate and has_detail):
        return {
            "query_shape": "mixed",
            "output_mode": "mixed",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Usuário pediu resumo e detalhes.",
        }

    if has_lookup or re.search(r"\b\d{5,}\b", text):
        return {
            "query_shape": "lookup",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Usuário parece buscar um registro específico.",
        }

    if has_distinct:
        return {
            "query_shape": "distinct",
            "output_mode": "rows",
            "limit": 200,
            "reason": "Usuário pediu lista de valores distintos.",
        }

    if has_detail:
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 20),
            "reason": "Usuário pediu registros individuais.",
        }

    if has_aggregate:
        return {
            "query_shape": "aggregate",
            "output_mode": "summary",
            "limit": 0,
            "reason": "Usuário pediu métrica agregada.",
        }

    # Fallback deliberado:
    # Se a pergunta não pede explicitamente total/percentual/agregação,
    # preferimos detalhe para evitar o comportamento antigo de agregar tudo.
    return {
        "query_shape": "detail",
        "output_mode": "detail",
        "limit": _detect_requested_limit(text, 20),
        "reason": "Fallback: sem pedido explícito de agregação, retornar detalhes.",
    }


def _build_query_shape_contract(query_shape: str, output_mode: str, sample_size: int, detail_limit: int) -> str:
    if query_shape == "aggregate":
        return """
Tipo de consulta inferido: aggregate

Use quando o usuário pedir total, quantidade, percentual, proporção, contagem ou indicador geral.

Regras:
- Retorne apenas métricas agregadas no SELECT final.
- Não liste pacientes ou atendimentos individuais.
- Métricas recomendadas:
  - total_registros
  - total_pacientes_unicos
  - positivos
  - provaveis
  - negativos
  - pos_operatorios
  - percentual_positivos
  - percentual_provaveis
  - percentual_negativos
"""

    if query_shape == "detail":
        return f"""
Tipo de consulta inferido: detail

Use quando o usuário pedir lista, casos, registros, pacientes, atendimentos, prontuários ou evidências individuais.

Regras obrigatórias:
- NÃO use COUNT, SUM, AVG, MIN, MAX, GROUP BY ou HAVING no SELECT final.
- NÃO transforme listagem em contagem.
- Retorne linhas individuais.
- Inclua, sempre que possível:
  - id_atendimento
  - id_paciente
  - nome_paciente
  - data_atendimento
  - classificacao
  - score
  - campo_origem
  - termo_detectado
  - trecho_evidencia
  - cid_codigo_txt
  - flg_cirurgica
  - clinica
  - regional
  - uf
  - municipio
  - nome_profissional
- Se precisar limitar, use row_number() em CTE chamada amostra/detalhes
  e filtre rn <= {detail_limit}.
"""

    if query_shape == "sample":
        return f"""
Tipo de consulta inferido: sample

Use quando o usuário pedir amostra, exemplos ou algumas linhas.

Regras obrigatórias:
- Ignore agregações.
- NÃO use COUNT, SUM, AVG, MIN, MAX, GROUP BY ou HAVING.
- Retorne apenas linhas individuais de amostra.
- A amostra deve ser integral, isto é, com colunas úteis do registro,
  evidência clínica e identificação do atendimento/paciente.
- Para limitar a amostra, use row_number() em uma CTE chamada amostra
  e filtre rn <= {sample_size}.
- Não use LIMIT.
"""

    if query_shape == "lookup":
        return f"""
Tipo de consulta inferido: lookup

Use quando o usuário buscar um paciente, atendimento, profissional ou registro específico.

Regras obrigatórias:
- NÃO use agregações.
- NÃO use COUNT, SUM, AVG, MIN, MAX, GROUP BY ou HAVING.
- Retorne registros individuais compatíveis com os critérios da pergunta.
- Inclua evidências clínicas e campos de identificação.
- Se precisar limitar, use row_number() e filtre rn <= {detail_limit}.
"""

    if query_shape == "breakdown":
        return """
Tipo de consulta inferido: breakdown

Use quando o usuário pedir agrupamento por clínica, regional, UF, município,
profissional, especialidade, classificação ou outro recorte.

Regras:
- Use GROUP BY somente nas dimensões pedidas ou claramente necessárias.
- Retorne uma linha por grupo.
- Métricas recomendadas:
  - total_registros
  - total_pacientes_unicos
  - positivos
  - provaveis
  - negativos
  - pos_operatorios
  - percentual_positivos
  - percentual_provaveis
  - percentual_negativos
- Não liste pacientes individuais, exceto se o usuário também pedir detalhes.
"""

    if query_shape == "timeline":
        return """
Tipo de consulta inferido: timeline

Use quando o usuário pedir evolução temporal, mês a mês, dia a dia,
semana a semana ou tendência.

Regras:
- Use date_trunc conforme a granularidade pedida:
  - mensal: date_trunc('month', data_atendimento)
  - semanal: date_trunc('week', data_atendimento)
  - diária: data_atendimento
- Retorne uma linha por período.
- Inclua métricas de volume e classificação.
- Ordene cronologicamente.
"""

    if query_shape == "mixed":
        return f"""
Tipo de consulta inferido: mixed

Use quando o usuário pedir resumo + detalhes, total + exemplos ou contagem + casos.

Regras:
- Separe agregações e linhas detalhadas em CTEs diferentes.
- CTE recomendada:
  base -> texto_normalizado -> features -> score_calc -> classificado -> resumo -> detalhes
- No SELECT final, pode usar CROSS JOIN entre resumo e detalhes.
- Limite as linhas detalhadas com row_number() e filtre rn <= {detail_limit}.
"""

    if query_shape == "distinct":
        return """
Tipo de consulta inferido: distinct

Use quando o usuário pedir quais valores existem, valores únicos,
clínicas, regionais, profissionais ou categorias sem repetir.

Regras:
- Use SELECT DISTINCT quando o usuário pedir apenas a lista.
- Só use COUNT/GROUP BY se o usuário pedir quantidade por valor.
- Retorne a coluna solicitada de forma limpa, sem registros duplicados.
"""

    return f"""
Tipo de consulta inferido: {query_shape}
Siga a intenção da pergunta do usuário.
"""


def _sql_has_aggregation(sql: str) -> bool:
    sql_lower = _normalize_text_intent(sql)

    for pattern in AGGREGATION_REGEXES:
        if re.search(pattern, sql_lower, flags=re.IGNORECASE):
            return True

    return False


def _semantic_validate_sql(sql: str, query_shape: str, original_query: str) -> None:
    """
    Validação semântica simples para evitar que pedidos de linha/amostra
    sejam respondidos com SQL agregado.
    """
    sql_lower = _normalize_text_intent(sql)

    if not sql_lower:
        raise ValueError("SQL vazio gerado pelo LLM.")

    if re.search(r"\bselect\s+\*", sql_lower):
        raise ValueError("SQL inválido semanticamente: SELECT * não é permitido.")

    if query_shape in DETAIL_QUERY_SHAPES and _sql_has_aggregation(sql):
        raise ValueError(
            "A pergunta pede registros individuais/amostra/busca específica, "
            "mas o SQL gerado contém agregação. Regenere sem COUNT, SUM, AVG, "
            "MIN, MAX, GROUP BY ou HAVING."
        )

    if query_shape in DETAIL_QUERY_SHAPES and "id_atendimento" not in sql_lower:
        logger.warning(
            "SQL detalhado sem id_atendimento detectado. Não bloqueado, mas pode reduzir rastreabilidade."
        )

    if "id_especialidade = 661" not in sql_lower and "id_especialidade=661" not in sql_lower:
        raise ValueError("SQL inválido semanticamente: filtro obrigatório id_especialidade = 661 ausente.")


def _generate_sql(
    query: str,
    rag_context: str,
    output_mode: str,
    sample_size: int,
    hoje: str,
    query_shape: str = "detail",
    detail_limit: int = 20,
    extra_instruction: str = "",
) -> str:
    """
    Usa o LLM para gerar uma query SQL válida para o Athena com base na pergunta
    do usuário, no contrato de dados e nas regras clínicas do RAG.
    """
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
- Use agregações SOMENTE quando o usuário pedir total, quantidade, percentual,
  proporção, distribuição, agrupamento, ranking ou evolução temporal.
- Se o usuário pedir lista, casos, pacientes, atendimentos, exemplos, amostra
  ou evidências individuais, retorne linhas detalhadas, SEM agregações.
- Nunca transforme pedido de listagem/amostra em contagem agregada.

{CATARATA_SCHEMA}

Contrato de Execução:
1. Tabela: pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
2. Filtro fixo obrigatório: id_especialidade = 661{period_filter}
4. output_mode resolvido: '{output_mode}'
5. query_shape resolvido: '{query_shape}'

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

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )

    return _strip_markdown_sql(response.content)


def _repair_sql_for_semantics(
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

Regenere a consulta do zero respeitando o query_shape '{query_shape}'.
Se query_shape for detail, sample ou lookup, NÃO use agregações.
"""

    return _generate_sql(
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
    """
    Formata e estrutura os resultados brutos do Athena em payload padronizado.

    Diferença importante:
    - summary/aggregate: retorna resumo agregado.
    - detail/sample/lookup: retorna linhas individuais, sem summary artificial.
    - breakdown/timeline/distinct: retorna linhas agrupadas ou distintas como rows.
    - mixed: retorna summary + linhas individuais.
    """
    if not results:
        return {
            "execution_status": "success",
            "summary": {},
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": None,
            "limitations": ["Nenhum registro encontrado para os critérios informados."],
        }

    first = results[0]

    if query_shape == "aggregate" or output_mode == "summary":
        return {
            "execution_status": "success",
            "summary": _extract_summary(first, len(results)),
            "rows": [],
            "row_count": len(results),
            "truncated": False,
            "error": None,
            "limitations": [],
        }

    if query_shape in {"detail", "lookup"}:
        max_rows = detail_limit
        truncated = len(results) > max_rows
        rows_to_return = results[:max_rows]

        return {
            "execution_status": "success",
            "summary": {},
            "rows": [_compact_detail_row(row) for row in rows_to_return],
            "row_count": len(rows_to_return),
            "total_rows_returned_by_athena": len(results),
            "truncated": truncated,
            "error": None,
            "limitations": ["Resultados truncados."] if truncated else [],
        }

    if query_shape == "sample" or output_mode == "sample":
        max_rows = sample_size
        truncated = len(results) > max_rows
        rows_to_return = results[:max_rows]

        return {
            "execution_status": "success",
            "summary": {},
            "rows": [_compact_detail_row(row) for row in rows_to_return],
            "row_count": len(rows_to_return),
            "total_rows_returned_by_athena": len(results),
            "truncated": truncated,
            "error": None,
            "limitations": ["Amostra truncada."] if truncated else [],
        }

    if query_shape == "mixed":
        max_rows = detail_limit
        truncated = len(results) > max_rows
        rows_to_return = results[:max_rows]

        return {
            "execution_status": "success",
            "summary": _extract_summary(first, len(results)),
            "rows": [_compact_detail_row(row) for row in rows_to_return],
            "row_count": len(rows_to_return),
            "total_rows_returned_by_athena": len(results),
            "truncated": truncated,
            "error": None,
            "limitations": ["Resultados detalhados truncados."] if truncated else [],
        }

    if query_shape in {"breakdown", "timeline", "distinct"}:
        max_rows = detail_limit if detail_limit > 0 else 200
        truncated = len(results) > max_rows
        rows_to_return = results[:max_rows]

        return {
            "execution_status": "success",
            "summary": {},
            "rows": [_make_json_safe(row) for row in rows_to_return],
            "row_count": len(rows_to_return),
            "total_rows_returned_by_athena": len(results),
            "truncated": truncated,
            "error": None,
            "limitations": ["Resultados truncados."] if truncated else [],
        }

    # Fallback seguro: preserva os dados retornados.
    return {
        "execution_status": "success",
        "summary": {},
        "rows": [_make_json_safe(row) for row in results],
        "row_count": len(results),
        "truncated": False,
        "error": None,
        "limitations": [],
    }


def sql_analyst_expert(
    query: str,
    rag_context: str,
    output_mode: Optional[str] = None,
    sample_size: int = 5,
    hoje: str = "",
    retry_count: int = 0,
) -> dict:
    """
    Especialista SQL para o sistema Iris.

    Gera, valida e executa uma query Athena, retornando payload estruturado.

    Comportamento novo:
    - Se output_mode vier vazio, None ou genérico, o agente decide a intenção
      automaticamente.
    - Se o usuário pedir amostra/listagem/detalhes, o SQL não pode ser agregado.
    - Se o SQL for semanticamente incompatível com o pedido, o agente regenera
      antes de executar.
    - Se a execução no Athena falhar, tenta corrigir uma única vez.
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

    if query_shape == "sample":
        sample_size = intent["limit"] or sample_size

    logger.info(
        "SQL Analyst Expert iniciado | "
        f"query_shape={query_shape} | "
        f"output_mode={resolved_output_mode} | "
        f"sample_size={sample_size} | "
        f"detail_limit={detail_limit} | "
        f"reason={intent.get('reason')}"
    )

    try:
        sql = _generate_sql(
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
            _semantic_validate_sql(sql, query_shape, query)
        except ValueError as semantic_error:
            logger.warning(f"SQL rejeitado por validação semântica: {semantic_error}")

            sql = _repair_sql_for_semantics(
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
            _semantic_validate_sql(sql, query_shape, query)

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

    try:
        results = _execute_athena_query(sql)

        # Captura os dados brutos no contexto para uso pelo Agente Avaliador
        captured = athena_results_context.get([])
        athena_results_context.set(captured + [{"sql": sql, "results": results}])

        payload = _format_sql_result(
            results=results,
            output_mode=resolved_output_mode,
            sample_size=sample_size,
            query_shape=query_shape,
            detail_limit=detail_limit,
        )

        payload["sql"] = sql
        payload["query_shape"] = query_shape
        payload["output_mode"] = resolved_output_mode
        payload["intent_reason"] = intent.get("reason")

        logger.info(
            "SQL Analyst: Execução bem-sucedida | "
            f"query_shape={query_shape} | "
            f"row_count={payload.get('row_count')}"
        )

        return payload

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Erro ao executar SQL no Athena: {error_msg}")

        # Retry: tenta uma única vez com SQL corrigido pelo LLM
        if retry_count < 1:
            logger.info("SQL Analyst: Tentando corrigir SQL e reexecutar (retry 1)...")

            try:
                llm = get_chat_model_openai(temperature=0.0, model="gpt-4.1-mini")

                shape_contract = _build_query_shape_contract(
                    query_shape=query_shape,
                    output_mode=resolved_output_mode,
                    sample_size=sample_size,
                    detail_limit=detail_limit,
                )

                fix_prompt = f"""O SQL abaixo causou erro no AWS Athena/Presto.

Pergunta original:
{query}

query_shape:
{query_shape}

output_mode:
{resolved_output_mode}

Contrato do tipo de consulta:
{shape_contract}

SQL com erro:
{sql}

Erro:
{error_msg}

Corrija o SQL para ser compatível com Presto/Athena e respeitar o schema:
pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia

Regras obrigatórias:
- Sempre manter id_especialidade = 661.
- Nunca usar SELECT *.
- Se query_shape for detail, sample ou lookup, NÃO use COUNT, SUM, AVG, MIN, MAX,
  GROUP BY ou HAVING.
- Retorne APENAS o SQL corrigido, sem markdown e sem explicações.
"""

                fix_response = llm.invoke([{"role": "user", "content": fix_prompt}])
                fixed_sql = _strip_markdown_sql(fix_response.content)

                _semantic_validate_sql(fixed_sql, query_shape, query)
                validate_sql(fixed_sql)

                results = _execute_athena_query(fixed_sql)

                captured = athena_results_context.get([])
                athena_results_context.set(captured + [{"sql": fixed_sql, "results": results}])

                payload = _format_sql_result(
                    results=results,
                    output_mode=resolved_output_mode,
                    sample_size=sample_size,
                    query_shape=query_shape,
                    detail_limit=detail_limit,
                )

                payload["sql"] = fixed_sql
                payload["query_shape"] = query_shape
                payload["output_mode"] = resolved_output_mode
                payload["intent_reason"] = intent.get("reason")
                payload["limitations"] = payload.get("limitations", []) + [
                    "SQL foi auto-corrigido na execução."
                ]

                logger.info(
                    "SQL Analyst: Retry bem-sucedido | "
                    f"query_shape={query_shape} | "
                    f"row_count={payload.get('row_count')}"
                )

                return payload

            except Exception as retry_err:
                logger.error(f"SQL Analyst: Retry também falhou: {retry_err}")

                return {
                    "execution_status": "error",
                    "error": {
                        "type": "sql_executor_error",
                        "message": f"Original: {error_msg}. Retry: {retry_err}",
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

        return {
            "execution_status": "error",
            "error": {"type": "sql_executor_error", "message": error_msg},
            "summary": {},
            "rows": [],
            "row_count": 0,
            "sql": sql,
            "query_shape": query_shape,
            "output_mode": resolved_output_mode,
            "intent_reason": intent.get("reason"),
            "limitations": ["Execução SQL falhou."],
        }