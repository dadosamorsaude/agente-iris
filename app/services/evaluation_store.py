"""
Persistência das avaliações de acurácia da Iris no Supabase REST.

Fallback in-memory limitado por `deque(maxlen=MEMORY_STORE_CAPACITY)` para
evitar crescimento ilimitado em ambientes sem banco (Render free 512 MB).
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from datetime import datetime, timedelta, timezone

from app.core.clients import supabase_request
from app.core.config import settings

logger = logging.getLogger(__name__)

# Fallback in-memory com teto (último N) — evita OOM se Supabase cair.
MEMORY_STORE_CAPACITY = 200
_memory_store: deque[dict] = deque(maxlen=MEMORY_STORE_CAPACITY)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Escrita
# ──────────────────────────────────────────────────────────────────────────────


async def save_evaluation(
    user_id: str,
    question: str,
    response: str,
    raw_athena_data: list[dict],
    evaluation: dict,
) -> None:
    """Persiste o resultado de uma avaliação."""
    record = {
        "user_id": user_id,
        "created_at": _now_iso(),
        "question": question,
        "response": response,
        "raw_data": raw_athena_data,
        "score": evaluation.get("score", 0),
        "approved": evaluation.get("aprovado", False),
        "errors": evaluation.get("erros_encontrados", []),
        "justification": evaluation.get("justificativa", ""),
        "breakdown": {
            "precisao_factual": evaluation.get("precisao_factual", 0),
            "completude": evaluation.get("completude", 0),
            "interpretacao_clinica": evaluation.get("interpretacao_clinica", 0),
            "aplicacao_normativa": evaluation.get("aplicacao_normativa", 0),
        },
        "model": evaluation.get("model", settings.MODEL_NAME),
    }

    if settings.supabase_rest_url and settings.DATABASE_API_KEY:
        try:
            resp = await supabase_request(
                "POST", "evaluation_logs_iris", json_body=record,
            )
            if resp is not None:
                resp.raise_for_status()
                logger.info(f"Avaliação salva no Supabase | score={record['score']}")
                return
        except Exception as e:
            logger.warning(f"Falha ao salvar avaliação via REST, usando in-memory: {e}")

    _memory_store.append(record)
    logger.info(
        f"Avaliação salva in-memory | score={record['score']} | total={len(_memory_store)}"
    )


async def save_execution_log(
    job_id: str,
    session_id: str,
    conversation_id: str,
    original_input: str,
    result: dict,
) -> None:
    """Salva o log de execução estruturado na tabela judge_evaluations."""
    if not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        logger.warning("Supabase não configurado. Ignorando gravação de log de execução.")
        return

    try:
        final_answer = result.get("final_answer", result.get("output", ""))

        def clip(v, max_chars: int) -> str | None:
            if v is None:
                return None
            s = str(v)
            return s[:max_chars] + "..." if len(s) > max_chars else s

        meta_payload = {
            "timestamp": _now_iso(),
            "workflow_stage": result.get("workflow_stage"),
            "unit_of_analysis": result.get("unit_of_analysis"),
            "periodo": result.get("periodo"),
            "validated": result.get("validated", False),
            "has_data": result.get("has_data", False),
            "orchestration": result.get("orchestration", {}),
            "final_delivery_policy": result.get(
                "final_delivery_policy", "delivered_without_blocking"
            ),
        }

        judge_output = result.get("judge_output") or {}
        judge_score = float(result.get("judge_score") or 0.0)

        payload = {
            "workflow_id": "projeto-iris-backend",
            "execution_id": job_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "original_input": clip(original_input, 3000),
            "candidate_answer": clip(final_answer, 4000),
            "final_output": clip(final_answer, 4000),
            "analysis_type": result.get("analysis_type", ""),
            "rag_used": result.get("rag_used", False),
            "sql_used": result.get("sql_used", False),
            "row_count": result.get("executor_row_count", 0),
            "judge_passed": result.get("judge_passed", False),
            "judge_score": judge_score,
            "block_reason": judge_output.get("block_reason") or (
                result.get("errorType") if result.get("error") else None
            ),
            "issues": result.get("issues") or judge_output.get("issues") or [],
            "error": result.get("error", False),
            "error_type": result.get("errorType"),
            "metadata": meta_payload,
            "retry_count": result.get("retry_count", 0),
            "max_retries": 0,
            "score_final": judge_score * 100.0,
        }

        resp = await supabase_request(
            "POST", "judge_evaluations", json_body=payload,
        )
        if resp is not None:
            resp.raise_for_status()
            logger.info(f"Log de execução salvo | job_id={job_id}")
    except Exception as e:
        logger.error(f"Erro ao salvar log de execução: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Leitura
# ──────────────────────────────────────────────────────────────────────────────


async def get_evaluation_summary() -> dict:
    """Retorna o resumo agregado de todas as avaliações."""
    if settings.supabase_rest_url and settings.DATABASE_API_KEY:
        try:
            resp = await supabase_request(
                "GET",
                "evaluation_logs_iris",
                params={"select": "score,approved,created_at,errors"},
            )
            if resp is not None:
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    return _summary_from_rows(rows)
        except Exception as e:
            logger.error(f"Erro ao buscar resumo via REST, usando in-memory: {e}")

    return _summary_from_rows(list(_memory_store))


async def get_evaluation_history(limit: int = 20) -> list[dict]:
    """Retorna as últimas avaliações via REST."""
    if settings.supabase_rest_url and settings.DATABASE_API_KEY:
        try:
            resp = await supabase_request(
                "GET",
                "evaluation_logs_iris",
                params={"order": "created_at.desc", "limit": str(limit)},
            )
            if resp is not None:
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Erro ao buscar histórico via REST: {e}")

    return sorted(_memory_store, key=lambda x: x["created_at"], reverse=True)[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _top_errors(errors: list[str], n: int = 5) -> list[str]:
    return [err for err, _ in Counter(errors).most_common(n)]


def _summary_from_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "total_evaluations": 0,
            "avg_score": 0.0,
            "approved_rate": 0.0,
            "avg_score_last_7d": 0.0,
            "common_errors": [],
        }

    total = len(rows)
    scores = [r.get("score", 0) or 0 for r in rows]
    approved_count = sum(1 for r in rows if r.get("approved"))
    avg_score = sum(scores) / total

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    scores_7d = []
    for r in rows:
        ts = r.get("created_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts)[:19])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                scores_7d.append(r.get("score", 0) or 0)
        except Exception:
            continue
    avg_score_7d = sum(scores_7d) / len(scores_7d) if scores_7d else 0.0

    all_errors: list[str] = []
    for r in rows:
        if not r.get("approved") and r.get("errors"):
            errs = r["errors"]
            if isinstance(errs, list):
                all_errors.extend(errs)

    return {
        "total_evaluations": total,
        "avg_score": round(avg_score, 1),
        "approved_rate": round(100.0 * approved_count / total, 1),
        "avg_score_last_7d": round(avg_score_7d, 1),
        "common_errors": _top_errors(all_errors),
    }
