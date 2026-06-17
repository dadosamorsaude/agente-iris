"""
Endpoint interno para trigger da indexação de prontuários no Pinecone.

Protegido por AGENTE_API_KEY. Pode ser chamado por:
- GitHub Actions (cron diário às 02:00 BRT)
- Chamada manual para carga histórica
- Qualquer scheduler externo

Rotas:
    POST /internal/index-prontuarios        → D-1 (padrão)
    POST /internal/index-prontuarios/historico → Carga 90 dias
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.security import get_api_key
from app.services.prontuario_indexer import index_yesterday, index_historical_90_days, index_date_range

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["indexer"])


class IndexRangeRequest(BaseModel):
    start_date: str  # YYYY-MM-DD, inclusivo
    end_date: str    # YYYY-MM-DD, exclusivo


class IndexResponse(BaseModel):
    status: str
    indexed: int
    skipped: int
    errors: int
    total_fetched: int


@router.post(
    "/index-prontuarios",
    response_model=IndexResponse,
    summary="Indexa prontuários D-1 no Pinecone",
    description=(
        "Busca os prontuários do dia anterior no Athena, gera embeddings "
        "anonimizados e faz upsert no Pinecone. Chamada pelo cron diário."
    ),
)
async def index_prontuarios_d1(api_key: str = Depends(get_api_key)):
    """Trigger de indexação D-1 para uso pelo cron diário."""
    logger.info("Trigger de indexação D-1 recebido via endpoint.")
    try:
        result = await index_yesterday()
        return IndexResponse(status="ok", **result)
    except Exception as e:
        logger.error(f"Erro na indexação D-1: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na indexação D-1: {str(e)}",
        )


@router.post(
    "/index-prontuarios/historico",
    response_model=IndexResponse,
    summary="Carga histórica 90 dias no Pinecone",
    description=(
        "Carga histórica dos últimos 90 dias. Processa em janelas de 7 dias. "
        "Custo estimado: ~$26 USD. Executar apenas uma vez na ativação."
    ),
)
async def index_prontuarios_historico(api_key: str = Depends(get_api_key)):
    """Trigger de carga histórica completa dos últimos 90 dias."""
    logger.info("Trigger de carga histórica 90 dias recebido via endpoint.")
    try:
        result = await index_historical_90_days()
        return IndexResponse(status="ok", **result)
    except Exception as e:
        logger.error(f"Erro na carga histórica: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na carga histórica: {str(e)}",
        )


@router.post(
    "/index-prontuarios/range",
    response_model=IndexResponse,
    summary="Indexa prontuários em período customizado",
    description="Permite indexar um intervalo específico de datas. Útil para reprocessamento.",
)
async def index_prontuarios_range(
    body: IndexRangeRequest,
    api_key: str = Depends(get_api_key),
):
    """Indexa prontuários em um período customizado (start_date inclusivo, end_date exclusivo)."""
    logger.info(f"Indexação customizada: {body.start_date} → {body.end_date}")
    try:
        result = await index_date_range(
            start_date=body.start_date,
            end_date=body.end_date,
        )
        return IndexResponse(status="ok", **result)
    except Exception as e:
        logger.error(f"Erro na indexação customizada: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na indexação: {str(e)}",
        )
