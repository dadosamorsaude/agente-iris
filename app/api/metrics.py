"""
Endpoints de métricas de acurácia da Iris.

GET /metrics/summary  — Resumo agregado (taxa de aprovação, score médio, erros comuns)
GET /metrics/history  — Histórico de avaliações individuais
"""

from typing import Optional
from fastapi import APIRouter, Security, Query
from pydantic import BaseModel

from app.api.security import get_api_key
from app.services.evaluation_store import get_evaluation_summary, get_evaluation_history
from app.core.logger import logger

router = APIRouter(prefix="/metrics", tags=["metrics"])


# ──────────────────────────────────────────────────────────────────────────────
# Schemas de Resposta
# ──────────────────────────────────────────────────────────────────────────────

class MetricsSummaryResponse(BaseModel):
    total_evaluations: int
    avg_score: float
    approved_rate: float
    avg_score_last_7d: float
    common_errors: list[str]


class EvaluationRecord(BaseModel):
    id: Optional[int] = None
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    question: str
    score: int
    approved: bool
    errors: list[str]
    justification: Optional[str] = None
    breakdown: Optional[dict] = None


class MetricsHistoryResponse(BaseModel):
    evaluations: list[dict]
    total: int


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/summary", response_model=MetricsSummaryResponse)
async def metrics_summary(api_key: str = Security(get_api_key)):
    """
    Retorna o resumo agregado das avaliações de acurácia do agente.

    - `total_evaluations`: total de respostas avaliadas
    - `avg_score`: score médio geral (0-100)
    - `approved_rate`: percentual de respostas com score >= 70
    - `avg_score_last_7d`: score médio dos últimos 7 dias
    - `common_errors`: erros mais frequentes identificados pelo avaliador
    """
    logger.info("Requisição GET /metrics/summary")
    summary = await get_evaluation_summary()
    return MetricsSummaryResponse(**summary)


@router.get("/history", response_model=MetricsHistoryResponse)
async def metrics_history(
    limit: int = Query(default=20, ge=1, le=100, description="Número de avaliações a retornar"),
    api_key: str = Security(get_api_key),
):
    """
    Retorna o histórico das últimas avaliações individuais, ordenadas da mais recente para a mais antiga.

    Use `limit` para controlar quantos registros retornar (máx. 100).
    """
    logger.info(f"Requisição GET /metrics/history | limit={limit}")
    history = await get_evaluation_history(limit=limit)
    return MetricsHistoryResponse(evaluations=history, total=len(history))
