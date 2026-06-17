"""
Curadoria e geração de aprendizados (lessons) da Iris.

- Carrega lições curadas do Supabase REST
- Gera novas lições a partir de cada execução com base no veredito do Judge
- Persiste as lições no Supabase

A lógica de regras (`generate_lessons_from_execution`) é declarativa via
LESSON_RULES — cada regra define quando dispara e o conteúdo da lição.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from app.core.clients import supabase_request
from app.core.config import settings
from app.services.intent import detect_intent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Carregamento das lições curadas
# ──────────────────────────────────────────────────────────────────────────────

_EMPTY_LESSONS_JSON = (
    "{\n  \"tipo\": \"aprendizados_curados_do_projeto\",\n"
    "  \"uso\": \"Nenhum banco configurado para aprendizados.\",\n"
    "  \"regras_prioritarias\": []\n}"
)


async def load_curated_lessons(memory_key: str = "iris_catarata") -> str:
    """
    Busca via Supabase REST até 15 aprendizados ativos curados da Iris e formata
    como checklist de prompt.
    """
    if not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        return _EMPTY_LESSONS_JSON

    try:
        response = await supabase_request(
            "GET",
            "memoria_aprendizados_iris",
            params={
                "memory_key": f"eq.{memory_key}",
                "active": "eq.true",
                "select": "category,lesson,reason,confidence,usage_count",
                "order": "confidence.desc,usage_count.desc",
                "limit": "15",
            },
        )
        if response is None:
            return _EMPTY_LESSONS_JSON
        response.raise_for_status()
        rows = response.json()

        # Re-ordenação em Python para simular o CASE condicional da query antiga
        priority_categories = {
            'anti_alucinacao': 0, 'amostra_sem_agregacao': 0, 'amostra': 0,
            'evidencias': 0, 'rag_sql_handoff': 0, 'sql': 0, 'rag_clinico': 0,
            'formato_saida': 0, 'relatorio_metricas': 0, 'qualidade': 0,
        }

        def sort_key(r):
            cat = r.get("category", "")
            cat_priority = priority_categories.get(
                cat, 2 if cat.startswith('bom_padrao') else 1
            )
            return (
                cat_priority,
                -float(r.get("confidence", 0)),
                -int(r.get("usage_count", 0)),
            )

        rows.sort(key=sort_key)

        aprendizados = [{
            "categoria": str(r.get("category", "")),
            "regra": str(r.get("lesson", "")),
            "motivo": str(r.get("reason") or "")[:500],
            "confianca": float(r.get("confidence") or 0.0),
            "usos": int(r.get("usage_count") or 0),
        } for r in rows]

        regras_prioritarias = [
            f"{item['categoria']}: {item['regra']}"
            for item in aprendizados if item["confianca"] >= 0.85
        ][:10]

        memory_context = {
            "tipo": "aprendizados_curados_do_projeto",
            "uso": [
                "Use como checklist operacional preventivo.",
                "Não copie respostas antigas.",
                "Não substitui RAG.",
                "Não substitui SQL.",
                "Não invente dados ausentes.",
                "Em conflito entre memória e dados/RAG, dados e RAG vencem.",
            ],
            "regras_prioritarias": regras_prioritarias,
            "aprendizados": aprendizados,
        }
        return json.dumps(memory_context, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.warning(f"Falha ao carregar aprendizados via REST: {e}")
        return _EMPTY_LESSONS_JSON


# ──────────────────────────────────────────────────────────────────────────────
# Geração de lições a partir de uma execução — tabela declarativa de regras
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionFacts:
    """Snapshot dos fatos relevantes de uma execução, derivado uma única vez."""
    original_input: str
    analysis_type: str
    final_answer: str
    error: bool
    error_type: str | None
    error_message: str | None
    judge_failed: bool
    judge_passed: bool | None
    judge_score: float | None
    rag_used: bool
    sql_used: bool
    is_sample: bool
    is_evidence: bool
    is_aggregation: bool
    judge_text: str

    def critique_has(self, *terms: str) -> bool:
        return any(t in self.judge_text for t in terms)

    def answer_has(self, *terms: str) -> bool:
        return any(t in self.final_answer for t in terms)


@dataclass(frozen=True)
class LessonRule:
    category: str
    lesson: str
    confidence: float
    when: Callable[[ExecutionFacts], bool]
    reason: Callable[[ExecutionFacts], str]


_HALLUCINATION_TERMS = (
    'inventou', 'invenção', 'invencao', 'alucinacao', 'alucinação',
    'sem base', 'nao sustentado', 'não sustentado',
    'dados nao encontrados', 'dados não encontrados',
)
_AGGREGATION_WORDS = (
    'percentual', 'porcentagem', 'total', 'resumo', 'distribuicao', 'distribuição',
)
_AGGREGATION_CRITIQUE = (
    'agregacao indevida', 'agregação indevida', 'agregado', 'consolidado',
)
_RAG_TERMS = (
    'rag', 'contexto clinico', 'contexto clínico',
    'regua', 'régua', 'classificacao', 'classificação',
)
_INCOMPLETENESS_TERMS = (
    'incompleto', 'faltou metrica', 'faltou métrica',
    'sem percentual', 'sem total',
    'sem estratificacao', 'sem estratificação',
)
_FORMAT_TERMS = (
    'json invalido', 'json inválido', 'formato invalido', 'formato inválido',
    'nao retornou json', 'não retornou json',
)


LESSON_RULES: list[LessonRule] = [
    LessonRule(
        category="erro_recorrente",
        lesson=(
            "Quando ocorrer um erro recorrente, tratar o erro antes de "
            "responder ao usuário."
        ),
        confidence=0.75,
        when=lambda f: f.error and bool(f.error_type),
        reason=lambda f: f.error_message or f.error_type or "",
    ),
    LessonRule(
        category="sql",
        lesson=(
            "Validar SQL de forma determinística antes de executar, "
            "respeitando sintaxe Athena/Presto e tipos compatíveis."
        ),
        confidence=0.90,
        when=lambda f: f.error_type == "sql_executor_error",
        reason=lambda f: "Execução SQL falhou ou foi rejeitada pelo executor.",
    ),
    LessonRule(
        category="rag_sql_handoff",
        lesson=(
            "Nunca chamar SQL sem rag_context clínico real retornado pelo RAG."
        ),
        confidence=0.95,
        when=lambda f: f.error_type == "missing_rag_context",
        reason=lambda f: "O contexto RAG estava ausente, genérico ou inválido.",
    ),
    LessonRule(
        category="qualidade",
        lesson=(
            "Quando o Judge pontuar baixo ou reprovar, revisar aderência aos "
            "dados, completude, ausência de invenção e formato final."
        ),
        confidence=0.80,
        when=lambda f: f.judge_failed,
        reason=lambda f: f"judge_passed={f.judge_passed}; judge_score={f.judge_score}",
    ),
    LessonRule(
        category="amostra",
        lesson=(
            "Quando o usuário pedir amostra, retornar poucas linhas individuais "
            "completas, sem agregações, percentuais ou resumo consolidado."
        ),
        confidence=0.95,
        when=lambda f: f.judge_failed and f.is_sample,
        reason=lambda f: f"Pedido de amostra teve baixa avaliação do Judge. judge_score={f.judge_score}",
    ),
    LessonRule(
        category="amostra_sem_agregacao",
        lesson=(
            "Em modo amostra, não consolidar resultados; mostrar registros reais "
            "com id_atendimento, id_paciente, data, classificação, score e evidência."
        ),
        confidence=0.97,
        when=lambda f: (
            f.judge_failed and f.is_sample
            and (f.answer_has(*_AGGREGATION_WORDS) or f.critique_has(*_AGGREGATION_CRITIQUE))
        ),
        reason=lambda f: "O Judge indicou ou o texto final sugere uso de agregações em uma solicitação de amostra.",
    ),
    LessonRule(
        category="evidencias",
        lesson=(
            "Quando o usuário pedir evidências, incluir campo de origem, "
            "termo detectado e trecho textual sempre que disponíveis."
        ),
        confidence=0.90,
        when=lambda f: f.judge_failed and f.is_evidence,
        reason=lambda f: f"Pedido de evidências teve baixa avaliação do Judge. judge_score={f.judge_score}",
    ),
    LessonRule(
        category="anti_alucinacao",
        lesson=(
            "Nunca inventar totais, percentuais, pacientes, atendimentos, "
            "scores, classificações ou evidências ausentes no resultado SQL."
        ),
        confidence=0.98,
        when=lambda f: f.judge_failed and f.critique_has(*_HALLUCINATION_TERMS),
        reason=lambda f: "O Judge indicou possível resposta sem sustentação nos dados.",
    ),
    LessonRule(
        category="rag_clinico",
        lesson=(
            "Para perguntas substantivas, usar o RAG como régua clínica antes "
            "do SQL e não substituir a régua por memória ou exemplos anteriores."
        ),
        confidence=0.92,
        when=lambda f: f.judge_failed and f.critique_has(*_RAG_TERMS),
        reason=lambda f: "O Judge indicou problema relacionado ao uso do RAG ou da classificação clínica.",
    ),
    LessonRule(
        category="relatorio_metricas",
        lesson=(
            "Em pedidos de relatório, contagem ou distribuição, incluir total, "
            "estratificação, percentuais quando disponíveis e interpretação objetiva."
        ),
        confidence=0.90,
        when=lambda f: (
            f.judge_failed and f.is_aggregation and f.critique_has(*_INCOMPLETENESS_TERMS)
        ),
        reason=lambda f: "O Judge indicou incompletude em resposta agregada ou relatório.",
    ),
    LessonRule(
        category="formato_saida",
        lesson=(
            "O orquestrador deve retornar somente JSON válido no formato "
            "contratado, sem markdown, comentários ou texto fora do objeto."
        ),
        confidence=0.95,
        when=lambda f: f.judge_failed and f.critique_has(*_FORMAT_TERMS),
        reason=lambda f: "O Judge indicou problema de formato na saída.",
    ),
    LessonRule(
        category="bom_padrao",
        lesson=(
            "Para perguntas substantivas, o padrão RAG antes de SQL produz "
            "resposta mais confiável."
        ),
        confidence=0.85,
        when=lambda f: (
            f.rag_used and f.sql_used and not f.error
            and f.judge_score is not None and f.judge_score >= 0.85
        ),
        reason=lambda f: "Execução com RAG, SQL e boa avaliação do Judge.",
    ),
    LessonRule(
        category="bom_padrao_amostra",
        lesson=(
            "Em pedidos de amostra, o melhor padrão é retornar poucos registros "
            "individuais com evidência textual, sem resumo agregado."
        ),
        confidence=0.88,
        when=lambda f: (
            f.rag_used and f.sql_used and not f.error
            and f.judge_score is not None and f.judge_score >= 0.85
            and f.is_sample
        ),
        reason=lambda f: "Execução de amostra aprovada pelo Judge.",
    ),
]


def _build_facts(execution_data: dict) -> ExecutionFacts:
    original_input = execution_data.get("originalInput", "") or ""
    analysis_type = str(execution_data.get("analysis_type", "")).lower()
    final_answer = str(execution_data.get("final_answer", "")).lower()

    judge_output = execution_data.get("judge_output") or {}
    judge_passed = execution_data.get("judge_passed")
    judge_score_raw = execution_data.get("judge_score")
    try:
        judge_score = float(judge_score_raw) if judge_score_raw is not None else None
    except (TypeError, ValueError):
        judge_score = None
    low_score = judge_score is not None and judge_score < 0.75
    judge_failed = (judge_passed is False) or low_score

    judge_text = " ".join([
        str(judge_output),
        str(execution_data.get('issues', [])),
        str(execution_data.get('errorType', '')),
        str(execution_data.get('errorMessage', '')),
        analysis_type,
        original_input.lower(),
        final_answer,
    ]).lower()

    intent = detect_intent(original_input)
    is_sample = intent["sample_mode"] or "amostra" in analysis_type
    is_evidence = intent["wants_rows"] and not intent["sample_mode"]
    is_aggregation = intent["has_aggregation_intent"]

    return ExecutionFacts(
        original_input=original_input,
        analysis_type=analysis_type,
        final_answer=final_answer,
        error=execution_data.get("error") is True,
        error_type=execution_data.get("errorType"),
        error_message=execution_data.get("errorMessage"),
        judge_failed=judge_failed,
        judge_passed=judge_passed,
        judge_score=judge_score,
        rag_used=execution_data.get("rag_used") is True,
        sql_used=execution_data.get("sql_used") is True,
        is_sample=is_sample,
        is_evidence=is_evidence,
        is_aggregation=is_aggregation,
        judge_text=judge_text,
    )


def generate_lessons_from_execution(execution_data: dict) -> list[dict]:
    """
    Aplica LESSON_RULES sobre o resultado de uma execução e formata as lições
    no schema esperado pelo backend.
    """
    facts = _build_facts(execution_data)

    final_lessons: list[dict] = []
    for rule in LESSON_RULES:
        if not rule.when(facts):
            continue
        final_lessons.append({
            "memory_key": "iris_catarata",
            "category": rule.category,
            "lesson": rule.lesson,
            "reason": str(rule.reason(facts))[:900],
            "confidence": float(rule.confidence),
            "source_job_id": execution_data.get("job_id"),
            "source_session_id": execution_data.get("sessionId"),
            "source_analysis_type": execution_data.get("analysis_type"),
            "source_error_type": execution_data.get("errorType"),
        })

    return final_lessons


# ──────────────────────────────────────────────────────────────────────────────
# Persistência
# ──────────────────────────────────────────────────────────────────────────────


async def save_learned_lessons(lessons: list[dict]) -> None:
    """Grava (UPSERT) os novos aprendizados no Supabase via REST."""
    if not lessons or not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        return

    payload = [{
        "memory_key": l["memory_key"],
        "category": l["category"],
        "lesson": l["lesson"],
        "reason": l["reason"],
        "confidence": l["confidence"],
        "usage_count": 1,
        "active": True,
    } for l in lessons]

    try:
        response = await supabase_request(
            "POST",
            "memoria_aprendizados_iris",
            json_body=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates"},
        )
        if response is None:
            return
        response.raise_for_status()
        logger.info(f"Salvos {len(lessons)} aprendizados via REST no Supabase.")
    except Exception as e:
        logger.error(f"Erro ao salvar lições via REST: {e}")
