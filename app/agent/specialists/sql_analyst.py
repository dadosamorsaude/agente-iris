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
from app.services.intent import normalize_text, detect_grouped_lists
from app.tools.athena import query_athena_tool, validate_sql, athena_results_context
from app.services.llm import get_chat_model_openai
from app.core.config import settings

logger = logging.getLogger(__name__)

# Schema oficial da tabela de catarata do N8N
CATARATA_SCHEMA = """
Tabela principal: pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia (alias: fp)
Dialeto: Athena/Presto SQL

Colunas disponíveis em fp:
- id_paciente: bigint
- nome_paciente: string
- id_atendimento: bigint
- data_atendimento: date
- id_especialidade: bigint (informativo; a tabela ja vem pre-filtrada para oftalmologia)
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

Tabela de dimensao de pacientes (somente para enriquecer com CPF):
pdgt_amorsaude_inteligencia.dm_pacientes_amei (alias: dp)
- id_paciente: bigint (chave de join com fp.id_paciente)
- cpf_paciente: string (CPF em texto, exibir completo sem mascaramento)

Regra de JOIN para CPF:
- Use LEFT JOIN pdgt_amorsaude_inteligencia.dm_pacientes_amei dp ON dp.id_paciente = fp.id_paciente
- Selecione dp.cpf_paciente APENAS quando o usuario pedir caracterizacao/segregacao
  de pacientes ou listas detalhadas por grupo. Em consultas puramente agregadas,
  NAO faca o join (evita custo desnecessario).

Campos narrativos/textuais para busca clínica (narrative_fields):
- anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao,
  exame_solicitado, prescricao, posologia, obs_atend_oftalmo, atestado,
  cid_descricao_detalhada

Filtros obrigatórios SEMPRE:
- Nenhum. A tabela fl_prontuarios_oftalmologia ja vem filtrada para atendimentos
  de oftalmologia. NAO inclua filtro de id_especialidade — a coluna existe
  apenas para referencia.

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


def _mask_cpf(value: Any) -> Optional[str]:
    """Formata o CPF no formato 000.000.000-00 sem mascarar ou anonimizar.

    Aceita CPF com ou sem mascara, ignora caracteres nao numericos, lida com nulos
    e valores invalidos retornando None ou o CPF completo formatado.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)

    if len(digits) == 11:
        return f"{digits[0:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:11]}"

    return raw


def _classify_group(value: Any) -> Optional[str]:
    """Normaliza o valor da coluna 'classificacao' para uma chave canonica de grupo."""
    if value is None:
        return None
    norm = normalize_text(str(value))
    if not norm:
        return None
    if "pos" in norm and "operator" in norm:
        return "pos_operatorios"
    if "positivo" in norm:
        return "positivos"
    if "provavel" in norm or "provaveis" in norm:
        return "provaveis"
    if "negativo" in norm:
        return "negativos"
    # Mantem o valor normalizado como bucket extra para nao perder dados.
    return norm.replace(" ", "_")


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

    # Caracterizacao explicita: sempre mixed, sem truncar e com JOIN de CPF.
    if detect_grouped_lists(query):
        return {
            "query_shape": "mixed",
            "output_mode": "mixed",
            "limit": 0,
            "grouped_lists": True,
            "reason": "Usuario pediu caracterizacao/segregacao de pacientes por grupo clinico.",
        }

    if explicit_mode in {"detail", "detalhe", "detalhado", "sample", "amostra", "lookup"}:
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 0),
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
            "limit": _detect_requested_limit(text, 0),
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
            "limit": _detect_requested_limit(text, 0),
            "reason": "Pedido combina metrica e dados brutos.",
        }

    if wants_detail or re.search(r"\b\d{5,}\b", text):
        return {
            "query_shape": "detail",
            "output_mode": "detail",
            "limit": _detect_requested_limit(text, 0),
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
        "limit": _detect_requested_limit(text, 0),
        "reason": "Fallback simples: preservar dados brutos.",
    }


def _build_query_shape_contract(
    query_shape: str,
    output_mode: str,
    sample_size: int,
    detail_limit: int,
    grouped_lists: bool = False,
) -> str:
    limit_instruction = (
        f"- Se precisar limitar registros brutos, prefira row_number() em CTE e filtre rn <= {detail_limit}."
        if detail_limit > 0
        else "- Não limite o número de registros retornados, traga todos os resultados solicitados."
    )

    grouped_block = ""
    if grouped_lists:
        grouped_block = """
MODO CARACTERIZACAO DE PACIENTES (obrigatorio neste pedido):
- O usuario pediu caracterizacao/segregacao de pacientes em grupos clinicos.
- Voce DEVE gerar uma unica consulta CTE-based que produza linhas individuais (uma linha por paciente x atendimento) com a coluna `classificacao` materializada como string explicita, usando exatamente os rotulos: 'positivo', 'provavel', 'negativo' ou 'pos_operatorio' (em minusculas, sem acento).
- Os limiares e regras de score que definem cada classificacao devem vir do rag_context (treinamento_ia_catarata). Construa o score em CTEs progressivas (base -> texto_normalizado -> features -> score_calc -> classificado).
- O SELECT final DEVE incluir EXATAMENTE estas colunas, nesta ordem:
    id_paciente,
    nome_paciente,
    dp.cpf_paciente AS cpf_paciente,
    id_atendimento,
    data_atendimento,
    clinica,
    regional,
    nome_profissional,
    classificacao,
    score,
    termo_detectado,
    trecho_evidencia,
    lateralidade
- `lateralidade` deve ser identificada a partir do prontuário (dos campos narrativos/textuais), retornando 'OD', 'OE', 'AO' ou null.
- `termo_detectado` deve ser construido via expressao logica (CASE/COALESCE) que aponta qual termo do rag_context ativou a classificacao.
- `trecho_evidencia` deve usar regexp_extract sobre o campo narrativo que casou (limite implicito de tamanho controlado em Python; nao trunque no SQL).
- LEFT JOIN obrigatorio: `LEFT JOIN pdgt_amorsaude_inteligencia.dm_pacientes_amei dp ON dp.id_paciente = fp.id_paciente` para trazer cpf_paciente.
- Use alias `fp` para `pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia`. NAO adicione filtro de id_especialidade — a tabela ja vem pre-filtrada para oftalmologia.
- NAO use LIMIT, NAO use SELECT *, NAO use row_number para cortar grupos. Traga TODAS as linhas de TODOS os grupos.
- Nao filtre por classificacao no SELECT final, exceto se o usuario pediu explicitamente apenas um grupo.
"""

    return f"""
Preferencia inicial de consulta: {query_shape}
Modo de saida preferido: {output_mode}
Limite preferido para linhas detalhadas: {detail_limit if detail_limit > 0 else 'Nenhum'}
Caracterizacao de pacientes solicitada: {'sim' if grouped_lists else 'nao'}

Use a pergunta do usuario como fonte principal de decisao. Voce pode retornar:
- aggregate: metricas, totais, percentuais, rankings, recortes ou series temporais.
- detail: registros brutos/individuais, amostras integrais, pacientes, atendimentos, evidencias.
- mixed: metricas e registros brutos juntos quando isso responder melhor ao pedido.

Regras de liberdade controlada:
- Se o usuario pedir dados brutos, registros, casos, pacientes, atendimentos, amostra ou evidencias, preserve linhas individuais.
- Para registros individuais, inclua SEMPRE os campos textuais/narrativos: anamnese, conduta, hipotese_diagnostica, observacao, orientacao, solicitacao, prescricao, exame_solicitado, obs_atend_oftalmo, cid_descricao_detalhada.
- Para registros individuais, inclua SEMPRE explicitamente os identificadores reais e corretos do banco de dados: id_paciente, nome_paciente, id_atendimento, data_atendimento, id_profissional, nome_profissional, id_clinica, clinica. NUNCA omita, invente ou use IDs fictícios/alucinados.
- Para metricas, use agregacoes livremente, mas mantenha nomes de colunas claros.
- Para respostas mistas, separe resumo e detalhes em CTEs e retorne ambos no resultado final.
{limit_instruction}
- Nao use LIMIT.
{grouped_block}"""


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

    # A tabela fl_prontuarios_oftalmologia ja vem filtrada para oftalmologia.
    # NAO exigimos mais id_especialidade = 661 — e ate desencorajado, para evitar
    # filtros redundantes que limitem indevidamente os dados.
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
    grouped_lists: bool = False,
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
        grouped_lists=grouped_lists,
    )

    llm = get_chat_model_openai(temperature=0.0, model=settings.MODEL_NAME_SQL)

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
2. NAO adicione filtro de id_especialidade — a tabela ja vem pre-filtrada para oftalmologia.{period_filter}
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
- Se um período foi extraído ({period_filter.strip()}), use rigorosamente o filtro temporal de data_atendimento fornecido. Não crie filtros temporais dinâmicos ou voláteis adicionais de data. Se nenhum período foi solicitado, não invente filtro temporal.
- Regras de Contagem Determinística e Íntegra:
  * Ao realizar contagens de atendimentos, consultas ou visitas, use obrigatoriamente COUNT(DISTINCT id_atendimento) (ou COUNT(1) se garantido que cada registro na CTE final represente um atendimento único).
  * Ao realizar contagens de pacientes únicos ou casos clínicos de pacientes, use obrigatoriamente COUNT(DISTINCT id_paciente).
  * Nunca use COUNT(*) de forma genérica se houver chance de duplicar ou distorcer contagens.
  * Para contagens categorizadas por classificação clínica (Positivo, Provável, Negativo), os limiares e expressões de cálculo de score/classificação devem seguir exatamente as definições do RAG (rag_context).
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
    grouped_lists: bool = False,
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
        grouped_lists=grouped_lists,
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
    grouped_lists: bool = False,
) -> str:
    """Tenta corrigir SQL que falhou na execução do Athena."""
    llm = get_chat_model_openai(temperature=0.0, model=settings.MODEL_NAME_SQL)

    shape_contract = _build_query_shape_contract(
        query_shape=query_shape,
        output_mode=output_mode,
        sample_size=sample_size,
        detail_limit=detail_limit,
        grouped_lists=grouped_lists,
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


def extract_laterality(row: dict) -> Optional[str]:
    """Extrai lateralidade ('OD', 'OE', 'AO' ou None) a partir de termos e evidências no registro."""
    term = str(row.get("termo_detectado") or row.get("termo") or "")
    evidence = str(row.get("trecho_evidencia") or row.get("evidencia") or "")
    
    narrative_text = ""
    narrative_keys = [
        'anamnese', 'conduta', 'hipotese_diagnostica', 'observacao',
        'orientacao', 'solicitacao', 'exame_solicitado', 'prescricao',
        'posologia', 'obs_atend_oftalmo', 'atestado', 'cid_descricao_detalhada'
    ]
    for k in narrative_keys:
        if k in row:
            narrative_text += " " + str(row[k] or "")
            
    text_lower = f"{term} {evidence} {narrative_text}".lower()
    text_lower = re.sub(r"\s+", " ", text_lower)
    
    # Busca por padrões bilaterais
    if re.search(r"\b(ambos\s+os\s+olhos|bilateral|ambos\s+olhos)\b", text_lower) or re.search(r"\ba\.o\.(?![a-zA-Z0-9])", text_lower):
        return "AO"
        
    # Busca por AO caso-sensitivo (para evitar casar com a preposição "ao" em minúsculo)
    raw_text = f"{term} {evidence} {narrative_text}"
    if re.search(r"\bAO\b", raw_text):
        return "AO"
    
    has_od = re.search(r"\b(od|olho\s+direito|olho\s+dir)\b", text_lower) or re.search(r"\bo\.d\.(?![a-zA-Z0-9])", text_lower)
    has_oe = re.search(r"\b(oe|olho\s+esquerdo|olho\s+esq)\b", text_lower) or re.search(r"\bo\.e\.(?![a-zA-Z0-9])", text_lower)
    
    if has_od and has_oe:
        return "AO"
    elif has_od:
        return "OD"
    elif has_oe:
        return "OE"
        
    return None


def _compact_detail_row(row: dict) -> dict:
    safe_row = _make_json_safe(row)

    raw_cpf = (
        row.get("cpf_paciente")
        or row.get("cpf")
        or row.get("num_cpf")
    )
    masked_cpf = _mask_cpf(raw_cpf)

    raw_lateralidade = row.get("lateralidade")
    if not raw_lateralidade:
        raw_lateralidade = extract_laterality(row)

    return {
        "estrato": row.get("estrato") or row.get("classificacao") or row.get("classe"),
        "classificacao": row.get("classificacao") or row.get("estrato") or row.get("classe"),
        "id_atendimento": row.get("id_atendimento"),
        "id_paciente": row.get("id_paciente"),
        "nome_paciente": row.get("nome_paciente"),
        "cpf_paciente": masked_cpf,
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
        "lateralidade": raw_lateralidade,
        "raw": safe_row,
    }


def _group_rows_by_classification(rows: list[dict]) -> dict:
    """Agrupa linhas por classificacao clinica canonica.

    Retorna um dict com chaves estaveis para os 4 grupos principais
    (positivos, provaveis, negativos, pos_operatorios) sempre presentes
    como lista (possivelmente vazia), mais quaisquer buckets extras
    encontrados no resultado.
    """
    buckets: dict[str, list] = {
        "positivos": [],
        "provaveis": [],
        "negativos": [],
        "pos_operatorios": [],
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        group = _classify_group(
            row.get("classificacao")
            or row.get("estrato")
            or row.get("classe")
        )
        if not group:
            buckets.setdefault("nao_classificados", []).append(row)
            continue
        buckets.setdefault(group, []).append(row)
    return buckets


def _format_sql_result(
    results: list[dict],
    output_mode: str,
    sample_size: int,
    query_shape: str = "detail",
    detail_limit: int = 20,
    grouped_lists: bool = False,
) -> dict:
    """Formata o retorno preservando dados brutos sempre que existirem.

    When grouped_lists=True, NUNCA trunca e produz `grouped_rows` agregando
    `rows` por classificacao clinica canonica, alem de `summary` com totais
    por grupo. CPF aparece sempre completo.
    """
    if not results:
        return {
            "execution_status": "success",
            "summary": {},
            "rows": [],
            "raw_rows": [],
            "grouped_rows": {} if not grouped_lists else {
                "positivos": [], "provaveis": [], "negativos": [], "pos_operatorios": [],
            },
            "row_count": 0,
            "total_rows_returned_by_athena": 0,
            "truncated": False,
            "error": None,
            "limitations": ["Nenhum registro encontrado para os criterios informados."],
        }

    if grouped_lists:
        max_rows = 0  # caracterizacao nunca trunca
    elif query_shape == "aggregate" and output_mode == "summary":
        max_rows = 0
    else:
        max_rows = detail_limit if detail_limit > 0 else 0

    if max_rows > 0:
        rows_to_return = results[:max_rows]
    else:
        rows_to_return = results

    truncated = len(results) > len(rows_to_return)

    if grouped_lists:
        # Caracterizacao: compactamos cada linha (mascara CPF, normaliza campos) e
        # agrupamos por classificacao. raw_rows preserva o registro original para auditoria,
        # mas o orquestrador deve usar grouped_rows.
        compact_rows = [_compact_detail_row(row) for row in rows_to_return]
        grouped = _group_rows_by_classification(compact_rows)
        safe_raw = [_make_json_safe(row) for row in rows_to_return]
        # Remove cpf cru de raw_rows para nao vazar CPF nao mascarado no log/judge.
        for r in safe_raw:
            if isinstance(r, dict) and "cpf_paciente" in r:
                r["cpf_paciente"] = _mask_cpf(r.get("cpf_paciente"))
        payload = {
            "execution_status": "success",
            "summary": {},
            "rows": compact_rows,
            "raw_rows": safe_raw,
            "grouped_rows": grouped,
            "group_counts": {k: len(v) for k, v in grouped.items()},
            "row_count": len(compact_rows),
            "total_rows_returned_by_athena": len(results),
            "truncated": False,
            "error": None,
            "limitations": [],
        }
        payload["summary"] = {
            "total_registros": len(results),
            "total_pacientes_unicos": len({
                r.get("id_paciente") for r in results if r.get("id_paciente") is not None
            }) or None,
            "positivos": len(grouped.get("positivos", [])),
            "provaveis": len(grouped.get("provaveis", [])),
            "negativos": len(grouped.get("negativos", [])),
            "pos_operatorios": len(grouped.get("pos_operatorios", [])),
        }
        return payload

    safe_rows = [_make_json_safe(row) for row in rows_to_return]
    # Mesmo fora de caracterizacao, se cpf_paciente aparecer (algum outro modo
    # detalhado), mascaramos antes de devolver para o agente/judge.
    for r in safe_rows:
        if isinstance(r, dict) and "cpf_paciente" in r:
            r["cpf_paciente"] = _mask_cpf(r.get("cpf_paciente"))

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

    if result_str.startswith("Consulta invalida") or result_str.startswith("Erro ao acessar"):
        raise ValueError(result_str)

    if result_str in ("Nenhum resultado encontrado para esta consulta.", ""):
        return []

    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Falha ao parsear resultado da tool: {e}")
        raise ValueError(f"Resultado invalido da ferramenta Athena: {result_str[:200]}")


@traceable(name="sql_analyst_expert", as_type="chain")
async def sql_analyst_expert(
    query: str,
    rag_context: str,
    output_mode: Optional[str] = None,
    sample_size: int = 5,
    hoje: str = "",
) -> dict:
    """Especialista SQL para o sistema Iris."""
    hoje = hoje or date.today().strftime("%Y-%m-%d")

    intent = _resolve_query_shape(
        query=query,
        output_mode=output_mode,
        sample_size=sample_size,
    )

    query_shape = intent["query_shape"]
    resolved_output_mode = intent["output_mode"]
    grouped_lists = bool(intent.get("grouped_lists"))
    detail_limit = 0 if grouped_lists else (intent["limit"] or 20)

    logger.info(
        "SQL Analyst Expert iniciado | "
        f"query_shape={query_shape} | "
        f"output_mode={resolved_output_mode} | "
        f"sample_size={sample_size} | "
        f"detail_limit={detail_limit} | "
        f"grouped_lists={grouped_lists} | "
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
            grouped_lists=grouped_lists,
        )

        logger.info(f"SQL gerado:\n{sql}")

        try:
            sql = _semantic_validate_sql(sql, query_shape, query)
        except ValueError as semantic_error:
            logger.warning(f"SQL rejeitado por validacao semantica: {semantic_error}")

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
                grouped_lists=grouped_lists,
            )

            logger.info(f"SQL regenerado apos validacao semantica:\n{sql}")

            try:
                sql = _semantic_validate_sql(sql, query_shape, query)
            except ValueError as e:
                logger.error(f"Repair tambem falhou validacao semantica: {e}")

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
                    "limitations": [f"Repair semantico falhou: {e}"],
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
        logger.warning(f"SQL invalido: {e}")

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
                grouped_lists=grouped_lists,
            )

            payload["sql"] = current_sql
            payload["query_shape"] = query_shape
            payload["output_mode"] = resolved_output_mode
            payload["grouped_lists"] = grouped_lists
            payload["intent_reason"] = intent.get("reason")

            if attempt > 0:
                payload["limitations"] = payload.get("limitations", []) + [
                    f"SQL foi auto-corrigido na tentativa {attempt + 1}."
                ]

            logger.info(
                "SQL Analyst: Execucao bem-sucedida | "
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
                    grouped_lists=grouped_lists,
                )
                try:
                    current_sql = _semantic_validate_sql(current_sql, query_shape, query)
                    validate_sql(current_sql)
                except ValueError as ve:
                    logger.warning(f"SQL regenerado continuou invalido: {ve}")

    logger.error(f"SQL Analyst: Todas as {max_attempts} tentativas falharam")

    return {
        "execution_status": "error",
        "error": {
            "type": "sql_executor_error",
            "message": f"Todas as {max_attempts} tentativas falharam. Ultimo erro: {last_error}",
        },
        "summary": {},
        "rows": [],
        "row_count": 0,
        "sql": sql,
        "query_shape": query_shape,
        "output_mode": resolved_output_mode,
        "intent_reason": intent.get("reason"),
        "limitations": [
            "Execucao SQL falhou mesmo apos tentativa de autocorrecao."
        ],
    }