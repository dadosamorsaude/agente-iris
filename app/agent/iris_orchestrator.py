"""
Iris Orchestrator — Deep Agent Principal

Orquestrador do novo sistema Iris. Coordena os agentes especialistas e
implementa o fluxo de raciocínio profundo (Multi-Agent Deep Architecture):
  1. Loader de Aprendizados Curados
  2. Clinical RAG Expert Agent
  3. SQL Query Analyst & Executor Agent decide o formato da consulta
  4. Iris Synthesizer (LLM)
  5. Quality Judge como metrica assincrona
  6. Self-Learning & Persistent Logger
"""

import asyncio
import json
import logging
import uuid
from datetime import date
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.observability import flush_langsmith, get_langsmith_callbacks, traceable
from app.services.learning import load_curated_lessons, generate_lessons_from_execution, save_learned_lessons
from app.services.evaluation_store import save_execution_log
from app.services.intent import detect_intent, light_interaction
from app.agent.evaluator import evaluate_response
from app.agent.specialists.clinical_rag import clinical_rag_expert
from app.agent.specialists.sql_analyst import sql_analyst_expert
from app.services.memory import get_session_history, add_user_message, add_ai_message
from app.services.llm import get_chat_model_openai
from app.tools.athena import athena_results_context
from app.tools.rag import rag_results_context

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de Texto
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""
    content_type = getattr(content, "type", None)
    if content_type == "text":
        return str(getattr(content, "text", ""))
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt do Iris Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

IRIS_SYNTHESIZER_SYSTEM = """Você é a Iris, agente orquestrador principal do sistema de auditoria de cirurgias de catarata.

Você receberá:
- A pergunta original do usuário
- O contexto clínico (rag_context) com as regras da régua de catarata
- Os dados do banco de dados (sql_result), quando a pergunta exigir consulta
- Aprendizados preventivos do projeto (memory_context) como checklist operacional
- O plano de ferramentas escolhidas para a interação

## Uso dos Aprendizados do Projeto
Os aprendizados são um checklist preventivo. Use-os para evitar erros recorrentes já identificados.
Eles NÃO substituem o RAG. Eles NÃO substituem o SQL. Eles NÃO autorizam inventar dados.
Em conflito: dados reais do SQL vencem. Régua clínica do RAG vence. Aprendizados servem apenas como orientação de processo.
Não copie o texto dos aprendizados na resposta ao usuário.

## Regras de Resposta
1. Saudação simples: responda diretamente sem invocar dados: "Olá, eu sou a Iris, em que posso te ajudar?"
2. Perguntas substantivas sobre catarata: use o rag_context como régua clínica e os dados do sql_result.
3. Nunca invente dados, totais, percentuais, pacientes, atendimentos, scores ou evidências.
4. Nunca exponha a arquitetura interna, nomes de agentes, steps técnicos ou SQL bruto ao usuário (exceto se explicitamente solicitado).
5. Se sql_result tiver erro ou estiver vazio: informe que não foi possível consultar os dados.
6. Se o resultado SQL trouxer registros individuais, preserve os detalhes relevantes na resposta.
7. Para relatório/contagem: inclua total, estratificação e percentuais quando disponíveis.
8. Se houver score, explique de forma simples os fatores que contribuíram.

## Formato de Saída OBRIGATÓRIO (JSON puro, sem markdown)
{
  "final_answer": "<resposta completa em texto para o usuário>",
  "analysis_type": "<contagem|listagem|distribuicao|amostra|relatorio|comparacao|classificacao_direta|conceitual|saudacao|erro>",
  "periodo": {"inicio": null, "fim_exclusivo": null},
  "unit_of_analysis": "<id_atendimento|id_paciente|nao_aplicavel>",
  "error": false,
  "errorType": null,
  "errorMessage": null,
  "rag_used": false,
  "sql_used": false,
  "user_asked_for_sql": false
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _truncate(text: Any, max_chars: int = 2500) -> str:
    t = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return t[:max_chars] + "..." if len(t) > max_chars else t


def _build_synthesizer_prompt(
    user_question: str,
    query_plan: dict,
    rag_context: str,
    sql_result: dict,
    memory_context: str,
    hoje: str = "",
) -> str:
    parts = [
        f"Pergunta: {user_question}",
        f"\nData de hoje: {hoje}",
        f"\nPlano de consulta: {json.dumps(query_plan or {}, ensure_ascii=False)}",
    ]

    if memory_context:
        parts.append(f"\nAprendizados do projeto (checklist preventivo):\n{_truncate(memory_context, 2000)}")

    if rag_context:
        parts.append(f"\nContexto Clínico RAG (régua de catarata):\n{_truncate(rag_context, 2500)}")

    if sql_result:
        exec_status = sql_result.get("execution_status", "unknown")
        if exec_status == "error":
            parts.append(f"\nResultado SQL: ERRO — {sql_result.get('error', {}).get('message', 'erro desconhecido')}")
        else:
            parts.append(f"\nResultado SQL:\n{_truncate(sql_result, 3000)}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Iris Synthesizer (chamada ao LLM)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="iris_synthesize", as_type="chain")
async def _iris_synthesize(
    user_question: str,
    query_plan: dict,
    rag_context: str,
    sql_result: dict,
    memory_context: str,
    history_messages: list,
    hoje: str = "",
    stream: bool = False,
) -> dict | AsyncGenerator:
    system = IRIS_SYNTHESIZER_SYSTEM
    user_content = _build_synthesizer_prompt(
        user_question, query_plan, rag_context, sql_result,
        memory_context, hoje
    )

    messages = [SystemMessage(content=system)] + history_messages + [HumanMessage(content=user_content)]
    llm = get_chat_model_openai()

    if stream:
        return llm.astream(messages)

    response = await llm.ainvoke(messages)
    raw = extract_text_from_content(getattr(response, "content", None)) or str(response)
    parsed = _safe_json(raw)

    if not parsed:
        logger.warning("Iris Synthesizer: resposta não é JSON válido, usando texto bruto.")
        return {
            "final_answer": raw,
            "analysis_type": "nao_aplicavel",
            "periodo": {"inicio": None, "fim_exclusivo": None},
            "unit_of_analysis": "nao_aplicavel",
            "error": False,
            "errorType": None,
            "errorMessage": None,
            "rag_used": bool(rag_context),
            "sql_used": bool(sql_result and sql_result.get("execution_status") != "error"),
            "user_asked_for_sql": False,
        }

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Iris Run Agent — Ponto de entrada principal
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="run_iris_agent", as_type="chain")
async def run_iris_agent(
    user_id: str,
    message: str,
    stream: bool = False,
    session_id: str | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    logger.info(f"Iris Deep Agent iniciado | user_id={user_id} | stream={stream}")

    if not message or not message.strip():
        yield "Por favor, digite uma mensagem."
        return

    job_id = str(uuid.uuid4())
    hoje = date.today().isoformat()
    effective_session = session_id or user_id
    effective_conversation = conversation_id or user_id

    athena_results_context.set([])
    rag_results_context.set([])

    # ── Fast-path via detect_intent (unificado) ──────────────────────────────
    intent = detect_intent(message)

    if intent.get("light_response"):
        await add_user_message(effective_session, message)
        await add_ai_message(effective_session, intent["light_response"])

        light_result = {
            "final_answer": intent["light_response"],
            "analysis_type": "saudacao",
            "periodo": {"inicio": None, "fim_exclusivo": None},
            "unit_of_analysis": "nao_aplicavel",
            "error": False,
            "errorType": None,
            "errorMessage": None,
            "rag_used": False,
            "sql_used": False,
            "user_asked_for_sql": False,
            "judge_passed": True,
            "judge_score": 1.0,
            "job_id": job_id,
            "sessionId": effective_session,
            "conversationId": effective_conversation,
            "originalInput": message,
            "validated": True,
            "has_data": False,
            "executor_row_count": 0,
            "executor_summary": {},
            "query_result": None,
            "final_delivery_policy": "delivered_without_blocking",
            "fast_path": "light_interaction",
        }
        asyncio.create_task(_post_execution(
            job_id, effective_session, effective_conversation,
            message, light_result, [], []
        ))

        yield intent["light_response"]
        return

    should_use_rag = not intent["is_simple"]
    should_use_sql = not intent["is_simple"] and (
        intent["has_aggregation_intent"] or intent["wants_rows"] or intent["sample_mode"]
    )

    query_plan = {
        "routing": "agent_decides_tools",
        "output_mode": intent.get("output_mode"),
        "sample_size": intent.get("sample_size") or 5,
        "rag_policy": "on_demand",
        "sql_policy": "on_demand",
        "rag_requested": should_use_rag,
        "sql_requested": should_use_sql,
    }

    history_messages = await get_session_history(effective_session)

    # ── PARALELIZAÇÃO: memory + RAG executam simultaneamente ─────────────────
    rag_context = ""
    captured_rag = []

    if should_use_rag:
        try:
            memory_context, rag_result = await asyncio.gather(
                load_curated_lessons(),
                clinical_rag_expert(message),
                return_exceptions=True,
            )

            if isinstance(memory_context, Exception):
                logger.error(f"load_curated_lessons falhou: {memory_context}")
                memory_context = '{"tipo":"aprendizados_curados_do_projeto","uso":"Erro ao carregar.","regras_prioritarias":[]}'

            if isinstance(rag_result, Exception):
                logger.error(f"RAG Expert falhou: {rag_result}")
                rag_context = ""
                captured_rag = []
            else:
                rag_context = rag_result
                captured_rag = rag_results_context.get([])
                rag_results_context.set(captured_rag)
                logger.info("RAG Expert concluído em paralelo com memory.")
        except Exception as e:
            logger.error(f"Erro no gather memory+RAG: {e}")
            memory_context = '{"tipo":"aprendizados_curados_do_projeto","uso":"Erro ao carregar.","regras_prioritarias":[]}'
    else:
        try:
            memory_context = await load_curated_lessons()
        except Exception as e:
            logger.error(f"load_curated_lessons falhou: {e}")
            memory_context = '{"tipo":"aprendizados_curados_do_projeto","uso":"Erro ao carregar.","regras_prioritarias":[]}'
        logger.info("RAG Expert ignorado: pergunta não exige contexto clínico.")

    rag_used = bool(rag_context)

    # ── SQL Analyst (async direto, sem thread) ───────────────────────────────
    sql_result = {}
    sql_used = False
    captured_athena = []

    if should_use_sql:
        try:
            sql_result = await sql_analyst_expert(
                message,
                rag_context,
                query_plan["output_mode"],
                query_plan["sample_size"],
                hoje,
                0,
            )
            captured_athena = athena_results_context.get([])
            athena_results_context.set(captured_athena)
            sql_used = sql_result.get("execution_status") != "error"
            logger.info(
                f"SQL Analyst concluído | status={sql_result.get('execution_status')} "
                f"| row_count={sql_result.get('row_count', 0)}"
            )
        except Exception as e:
            logger.error(f"SQL Analyst falhou: {e}")
            sql_result = {
                "execution_status": "error",
                "error": {"message": str(e)},
                "rows": [],
                "summary": {},
                "row_count": 0,
            }
    else:
        logger.info("SQL Analyst ignorado: pergunta não exige consulta de dados.")

    query_plan.update({
        "query_shape": sql_result.get("query_shape"),
        "output_mode": sql_result.get("output_mode") or query_plan["output_mode"],
        "intent_reason": sql_result.get("intent_reason"),
    })

    # ── Iris Synthesizer ──────────────────────────────────────────────────────
    final_answer = ""

    if stream:
        full_response_parts = []
        try:
            stream_gen = await _iris_synthesize(
                message, query_plan, rag_context, sql_result,
                memory_context, history_messages, hoje=hoje, stream=True
            )
            async for chunk in stream_gen:
                token = extract_text_from_content(getattr(chunk, "content", None))
                if token:
                    full_response_parts.append(token)
                    yield token

        except asyncio.CancelledError:
            logger.warning("Streaming Iris cancelado pelo cliente.")
            return
        except Exception as e:
            logger.exception("Iris Synthesizer falhou durante streaming")
            error_msg = "Não foi possível gerar uma resposta no momento. Tente novamente."
            yield error_msg
            full_response_parts = [error_msg]

        raw_streamed = "".join(full_response_parts)
        parsed = _safe_json(raw_streamed)
        if parsed:
            final_answer = parsed.get("final_answer", raw_streamed)
            synth_result = parsed
        else:
            final_answer = raw_streamed
            synth_result = {
                "final_answer": final_answer,
                "analysis_type": "nao_aplicavel",
                "error": False,
                "errorType": None,
                "errorMessage": None,
            }

    else:
        try:
            synth_result = await _iris_synthesize(
                message, query_plan, rag_context, sql_result,
                memory_context, history_messages, hoje=hoje, stream=False
            )
        except Exception as e:
            logger.exception("Iris Synthesizer falhou")
            synth_result = {
                "final_answer": "Não foi possível gerar uma resposta no momento. Tente novamente.",
                "analysis_type": "erro",
                "error": True,
                "errorType": "synthesizer_error",
                "errorMessage": str(e),
                "rag_used": rag_used,
                "sql_used": sql_used,
                "user_asked_for_sql": False,
            }

        final_answer = synth_result.get("final_answer", "Não foi possível gerar uma resposta.")

    # ── Pós-processamento comum ───────────────────────────────────────────────
    synth_result["rag_used"] = rag_used
    synth_result["sql_used"] = sql_used
    synth_result.update({
        "job_id": job_id,
        "sessionId": effective_session,
        "conversationId": effective_conversation,
        "originalInput": message,
        "validated": not synth_result.get("error", False),
        "has_data": sql_used and (sql_result.get("row_count", 0) > 0),
        "executor_row_count": sql_result.get("row_count", 0),
        "executor_summary": sql_result.get("summary", {}),
        "query_result": sql_result if sql_used else None,
        "final_delivery_policy": "delivered_without_blocking",
    })

    await add_user_message(effective_session, message)
    await add_ai_message(effective_session, final_answer)

    raw_athena = athena_results_context.get([])
    raw_rag = rag_results_context.get([])
    asyncio.create_task(_post_execution(
        job_id, effective_session, effective_conversation,
        message, synth_result, raw_athena, raw_rag
    ))

    if not stream:
        yield final_answer


# ─────────────────────────────────────────────────────────────────────────────
# Post-Execution: Auto-Learning + Logging (background)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="iris_post_execution", as_type="chain")
async def _post_execution(
    job_id: str,
    session_id: str,
    conversation_id: str,
    original_input: str,
    result: dict,
    raw_athena: list,
    raw_rag: list,
) -> None:
    try:
        if result.get("judge_output") is None and (raw_athena or raw_rag):
            evaluation = await evaluate_response(
                user_question=original_input,
                agent_response=result.get("final_answer", result.get("output", "")),
                raw_athena_data=raw_athena,
                rag_context=raw_rag,
            )
            judge_score = evaluation.get("overall_score")
            if isinstance(judge_score, (int, float)) and judge_score > 1:
                judge_score = judge_score / 100.0

            result["judge_output"] = evaluation
            result["judge_passed"] = evaluation.get("judge_passed")
            result["judge_score"] = judge_score
            result["issues"] = evaluation.get("issues", [])

        result.setdefault("final_delivery_policy", "delivered_without_blocking")

        lessons = generate_lessons_from_execution(result)
        if lessons:
            await save_learned_lessons(lessons)
            logger.info(f"Auto-Learning: {len(lessons)} lição(ões) salva(s).")

        await save_execution_log(
            job_id=job_id,
            session_id=session_id,
            conversation_id=conversation_id,
            original_input=original_input,
            result=result,
        )
    except Exception as e:
        logger.error(f"Erro no post-execution da Iris: {e}")
    finally:
        flush_langsmith()
