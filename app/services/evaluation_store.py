"""
Persistência das avaliações de acurácia da Iris no Supabase REST API.

Mantém histórico de avaliações e execuções no Supabase,
usando fallback para lista in-memory se DATABASE_URL/API_KEY não estiverem configurados.
"""

import json
import logging
from datetime import datetime
from collections import Counter

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Fallback in-memory quando não há banco configurado
_memory_store: list[dict] = []

def _get_headers() -> dict:
    if not settings.DATABASE_API_KEY:
        return {}
    return {
        "apikey": settings.DATABASE_API_KEY,
        "Authorization": f"Bearer {settings.DATABASE_API_KEY}",
        "Content-Type": "application/json"
    }


async def save_evaluation(
    user_id: str,
    question: str,
    response: str,
    raw_athena_data: list[dict],
    evaluation: dict,
) -> None:
    """
    Persiste o resultado de uma avaliação.
    Usa Supabase REST se disponível, caso contrário armazena em memória.
    """
    record = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
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
            url = f"{settings.supabase_rest_url}evaluation_logs"
            resp = httpx.post(url, headers=_get_headers(), json=record, timeout=10.0)
            resp.raise_for_status()
            logger.info(f"Avaliação salva no Supabase via REST | score={record['score']}")
            return
        except Exception as e:
            logger.warning(f"Falha ao salvar avaliação via REST, usando in-memory: {e}")

    # Fallback: in-memory (sem persistência entre reinicializações)
    _memory_store.append(record)
    logger.info(f"Avaliação salva in-memory | score={record['score']} | total={len(_memory_store)}")


async def save_execution_log(
    job_id: str,
    session_id: str,
    conversation_id: str,
    original_input: str,
    result: dict
) -> None:
    """
    Salva o log de execução clínica estruturado da Iris na tabela de auditoria judge_evaluations (Supabase).
    Garante fidelidade total com os campos configurados no logging.json.
    """
    if not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        logger.warning("Supabase não configurado. Ignorando gravação de log de execução.")
        return

    try:
        final_answer = result.get("final_answer", result.get("output", ""))
        
        # Limita caracteres nos campos textuais grandes conforme o Preparar Log do n8n
        def clip(v, max_chars):
            if v is None: return None
            s = str(v)
            return s[:max_chars] + "..." if len(s) > max_chars else s

        meta_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_stage": result.get("workflow_stage"),
            "unit_of_analysis": result.get("unit_of_analysis"),
            "periodo": result.get("periodo"),
            "validated": result.get("validated", False),
            "has_data": result.get("has_data", False),
            "orchestration": result.get("orchestration", {}),
            "final_delivery_policy": result.get("final_delivery_policy", "delivered_without_blocking"),
        }

        judge_output = result.get("judge_output") or {}

        # Payload mapeado 100% conforme a estrutura de colunas do N8N (logging.json)
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
            "judge_score": float(result.get("judge_score") or 0.0),
            "block_reason": judge_output.get("block_reason") or (result.get("errorType") if result.get("error") else None),
            "issues": result.get("issues") or judge_output.get("issues") or [],
            "error": result.get("error", False),
            "error_type": result.get("errorType"),
            "metadata": meta_payload,
            "retry_count": result.get("retry_count", 0),
            "max_retries": 0,
            "score_final": float(result.get("judge_score") or 0.0) * 100.0,  # converte de volta para escala 0-100 para o Metabase
        }

        url = f"{settings.supabase_rest_url}judge_evaluations"
        resp = httpx.post(url, headers=_get_headers(), json=payload, timeout=10.0)
        resp.raise_for_status()
        logger.info(f"Log de execução da Iris salvo com sucesso via REST na tabela 'judge_evaluations' | job_id={job_id}")
    except Exception as e:
        logger.error(f"Erro ao salvar log de execução via REST em 'judge_evaluations': {e}")



async def get_evaluation_summary() -> dict:
    """Retorna o resumo agregado de todas as avaliações trazendo e calculando os dados em Python."""
    if settings.supabase_rest_url and settings.DATABASE_API_KEY:
        try:
            url = f"{settings.supabase_rest_url}evaluation_logs?select=score,approved,created_at,errors"
            resp = httpx.get(url, headers=_get_headers(), timeout=10.0)
            resp.raise_for_status()
            rows = resp.json()

            if not rows:
                return _summary_from_memory()

            total = len(rows)
            scores = [r.get("score", 0) or 0 for r in rows]
            approved = [1 for r in rows if r.get("approved")]
            avg_score = sum(scores) / total

            from datetime import datetime, timedelta
            limit_7d = datetime.utcnow() - timedelta(days=7)
            
            scores_7d = []
            for r in rows:
                try:
                    # Supabase returns ISO format: 2026-05-26T14:06:08.123456+00:00
                    dt = datetime.fromisoformat(r["created_at"][:19]) 
                    if dt >= limit_7d:
                        scores_7d.append(r.get("score", 0) or 0)
                except:
                    pass
            avg_score_7d = sum(scores_7d) / len(scores_7d) if scores_7d else 0.0

            # Computa top erros apenas onde approved = false
            errors_list = []
            for r in rows:
                if not r.get("approved") and r.get("errors"):
                    errs = r["errors"]
                    if isinstance(errs, list):
                        errors_list.extend(errs)

            common_errors = _top_errors(errors_list)

            return {
                "total_evaluations": total,
                "avg_score": round(avg_score, 1),
                "approved_rate": round(100.0 * len(approved) / total, 1),
                "avg_score_last_7d": round(avg_score_7d, 1),
                "common_errors": common_errors,
            }

        except Exception as e:
            logger.error(f"Erro ao buscar resumo via REST, caindo no fallback in-memory: {e}")

    # Fallback: calcular de _memory_store
    return _summary_from_memory()


async def get_evaluation_history(limit: int = 20) -> list[dict]:
    """Retorna as últimas avaliações via REST."""
    if settings.supabase_rest_url and settings.DATABASE_API_KEY:
        try:
            url = f"{settings.supabase_rest_url}evaluation_logs?order=created_at.desc&limit={limit}"
            resp = httpx.get(url, headers=_get_headers(), timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Erro ao buscar histórico via REST: {e}")

    # Fallback in-memory
    return sorted(_memory_store, key=lambda x: x["created_at"], reverse=True)[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ──────────────────────────────────────────────────────────────────────────────

def _top_errors(errors: list[str], n: int = 5) -> list[str]:
    """Retorna os N erros mais frequentes."""
    return [err for err, _ in Counter(errors).most_common(n)]


def _summary_from_memory() -> dict:
    data = _memory_store
    if not data:
        return {"total_evaluations": 0, "avg_score": 0.0,
                "approved_rate": 0.0, "avg_score_last_7d": 0.0, "common_errors": []}

    scores = [d["score"] for d in data]
    approved = [d["approved"] for d in data]
    all_errors = [e for d in data for e in d.get("errors", [])]

    return {
        "total_evaluations": len(data),
        "avg_score": round(sum(scores) / len(scores), 1),
        "approved_rate": round(100 * sum(approved) / len(approved), 1),
        "avg_score_last_7d": round(sum(scores) / len(scores), 1),
        "common_errors": _top_errors(all_errors),
    }
