"""
Iris Orchestrator — React Agent
"""

import asyncio
import logging
import uuid
from datetime import date
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.prebuilt import create_react_agent

from app.core.observability import flush_langsmith, traceable
from app.services.intent import detect_intent
from app.agent.evaluator import evaluate_response
from app.services.learning import generate_lessons_from_execution, save_learned_lessons
from app.services.evaluation_store import save_execution_log
from app.services.memory import get_session_history, add_user_message, add_ai_message
from app.services.llm import get_chat_model_claude
from app.tools.athena import athena_results_context
from app.tools.rag import rag_results_context
from app.agent.tools import tools
from app.services.prontuario_indexer import index_batch_by_ids

logger = logging.getLogger(__name__)

IRIS_SYSTEM_PROMPT = """Você é a Iris, agente orquestrador principal do sistema de auditoria de cirurgias de catarata.

Você tem ferramentas à sua disposição para:
1. Buscar o contexto clínico (RAG) da régua de catarata.
2. Consultar dados estruturados do banco de dados (SQL).
3. Buscar aprendizados do projeto para evitar erros.

## Fluxo Obrigatório
- Quando a pergunta envolver regras ou métricas, use a ferramenta de contexto clínico ANTES de usar a ferramenta de SQL.
- Passe o resultado do contexto clínico como `rag_context` para a ferramenta SQL.
- Use a ferramenta de aprendizados se precisar de orientação adicional.

## Regras de Resposta
1. Saudação simples: responda diretamente sem invocar dados.
2. Nunca invente dados, totais, percentuais, pacientes, atendimentos, scores ou evidências.
3. Nunca exponha a arquitetura interna, nomes de agentes, steps técnicos ou SQL bruto ao usuário (exceto se explicitamente solicitado).
4. Se o SQL retornar erro ou vazio: informe que não foi possível consultar os dados.
5. Se o resultado SQL trouxer registros individuais, preserve os detalhes relevantes na resposta.
6. Para relatório/contagem: inclua total, estratificação e percentuais quando disponíveis.

## Diretrizes de Fidelidade Numérica e Integridade de Sessão
1. Fidelidade Numérica Absoluta: Transcreva os números gerados pela ferramenta SQL exatamente como foram retornados no banco de dados. Nunca arredonde, altere, resuma ou estime valores (por exemplo, se o resultado for 42, escreva '42', nunca 'cerca de 40' ou 'mais de 40').
2. Especificação da Métrica de Contagem: Sempre especifique e diferencie com clareza o número total de "atendimentos/consultas" e o número de "pacientes únicos" (por exemplo, 'X atendimentos referentes a Y pacientes únicos').
3. Menção de Período Temporal: Sempre informe claramente ao usuário qual o período de data_atendimento que foi considerado na contagem gerada (por exemplo, 'no período de DD/MM/AAAA a DD/MM/AAAA'), garantindo rastreabilidade e integridade.
4. Identificadores Reais: Exiba apenas identificadores reais de pacientes (id_paciente, nome_paciente, cpf_paciente) e atendimentos (id_atendimento) conforme retornados pela consulta SQL. É expressamente proibido alucinar CPFs ou IDs fictícios.
5. Consistência de Filtros em Histórico: Ao processar perguntas consecutivas dentro de uma mesma sessão, verifique o histórico para manter a consistência temporal (filtros de data) e outros filtros aplicados anteriormente (clínica, profissional), a menos que o usuário peça explicitamente para alterá-los.

## Modo Caracterização de Pacientes
Quando a tool SQL retornar `grouped_lists: true` e/ou `grouped_rows` populado:
1. A resposta DEVE conter (i) um resumo agregado com os totais por classificação clínica (positivos, prováveis, negativos, pós-operatórios) e (ii) a LISTA COMPLETA de pacientes de CADA grupo presente em `grouped_rows`. NUNCA omita pacientes nem trunque grupos sob argumento de tamanho.
2. Para cada paciente listado, mostre obrigatoriamente: id_paciente, nome_paciente, cpf_paciente, id_atendimento, data_atendimento, clinica, regional, nome_profissional, score, termo_detectado e trecho_evidencia.
3. CPF deve ser exibido exatamente como entregue pela tool (completo e formatado sem mascaramento).
4. Organize a resposta por grupo (uma seção por classificação), com o total do grupo no cabeçalho e a lista de pacientes em seguida.
5. Se um grupo estiver vazio, declare explicitamente "Nenhum paciente nesta classificação para o período/critério".

## Busca Semântica de Prontuários (processo interno)
1. Quando a pergunta envolver classificação clínica, identificação de casos ou termos oftalmológicos, derive INTERNAMENTE uma query clínica com base no léxico retornado pelo RAG (ex: 'facectomia indicação catarata LIO bilateral') e invoque `search_similar_records` ANTES de `analyze_and_execute_sql`.
2. Use os `ids_atendimento` retornados como filtro adicional `WHERE id_atendimento IN (...)` no SQL, ampliando o recall de casos com variações de terminologia.
3. Este processo é TOTALMENTE INTERNO. Nunca mencione ao usuário que uma busca vetorial foi realizada, nem exponha os IDs retornados ou o score de similaridade.
4. Fluxo obrigatório quando aplicável: `fetch_clinical_guidelines` → `search_similar_records` → `analyze_and_execute_sql`.
5. Se `search_similar_records` retornar lista vazia, prossiga normalmente com SQL via regex — a busca semântica é um complemento, não um bloqueio.
"""

def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    if getattr(content, "type", None) == "text":
        return str(getattr(content, "text", ""))
    return ""


@traceable(name="run_iris_agent", as_type="chain")
async def run_iris_agent(
    user_id: str,
    message: str,
    stream: bool = False,
    session_id: str | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    logger.info(f"Iris React Agent iniciado | user_id={user_id} | stream={stream}")

    if not message or not message.strip():
        yield "Por favor, digite uma mensagem."
        return

    # Guardrail de Entrada (Estratégia A) contra Prompt Injection
    from app.services.guardrails import is_message_safe
    is_safe, error_msg = await is_message_safe(message)
    if not is_safe:
        yield error_msg
        return

    job_id = str(uuid.uuid4())
    hoje = date.today().isoformat()
    effective_session = session_id or user_id
    effective_conversation = conversation_id or user_id

    athena_results_context.set([])
    rag_results_context.set([])

    # ── Fast-path via detect_intent ──────────────────────────────
    intent = detect_intent(message)

    if intent.get("light_response"):
        await add_user_message(effective_session, message)
        await add_ai_message(effective_session, intent["light_response"])

        light_result = {
            "final_answer": intent["light_response"],
            "analysis_type": "saudacao",
            "error": False,
            "rag_used": False,
            "sql_used": False,
            "judge_passed": True,
            "judge_score": 1.0,
            "job_id": job_id,
            "sessionId": effective_session,
            "conversationId": effective_conversation,
            "originalInput": message,
        }
        asyncio.create_task(_post_execution(
            job_id, effective_session, effective_conversation,
            message, light_result, [], []
        ))

        yield intent["light_response"]
        return

    # Histórico
    history_messages = await get_session_history(effective_session)

    # Injeta a data atual no system prompt: sem isso, o LLM responde com a data
    # da memoria de treinamento (ex.: "2025"). Mantemos o template como constante
    # e prefixamos o contexto temporal por execucao.
    hoje_dt = date.today()
    hoje_br = hoje_dt.strftime("%d/%m/%Y")
    runtime_prompt = (
        f"## Contexto Temporal (autoritativo)\n"
        f"- Data atual: {hoje_br} (ISO: {hoje}).\n"
        f"- Quando o usuario perguntar a data, ano ou periodo atual, responda EXATAMENTE com base nesta data. "
        f"Nao use a data da sua memoria de treinamento.\n"
        f"- Ao interpretar termos relativos (hoje, ontem, este mes, mes passado, ultimos N dias), "
        f"calcule a partir de {hoje_br}.\n\n"
        f"{IRIS_SYSTEM_PROMPT}"
    )

    from app.services.llm import get_chat_model_openai

    primary_llm = get_chat_model_claude()
    fallback_llm = get_chat_model_openai()
    llm = primary_llm.with_fallbacks([fallback_llm])
    react_agent = create_react_agent(llm, tools=tools, prompt=runtime_prompt)

    messages = history_messages + [HumanMessage(content=message)]

    final_answer = ""
    sql_used = False
    rag_used = False
    
    if stream:
        TOOL_ALIASES = {
            "analyze_and_execute_sql": "Análise e Consulta SQL",
            "fetch_clinical_guidelines": "Busca de Diretrizes (RAG)",
            "prontuario_search": "Busca de Histórico Médico",
        }

        event_queue = asyncio.Queue()
        active_tools = 0

        async def consume_stream():
            try:
                async for event in react_agent.astream_events({"messages": messages}, version="v2"):
                    await event_queue.put(event)
            except Exception as e:
                await event_queue.put(e)
            finally:
                await event_queue.put(None)

        stream_task = asyncio.create_task(consume_stream())

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Envia espaço Keep-Alive para evitar timeout de 100s no Render
                    yield " "
                    continue

                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event

                kind = event.get("event")

                if kind == "on_tool_start":
                    active_tools += 1
                    tool_name = event.get("name", "ferramenta")
                    alias = TOOL_ALIASES.get(tool_name, tool_name)
                    if active_tools == 1:
                        logger.info(f"Executando ferramenta: {tool_name}")
                        yield f"\n[⚙️ Pensando: Acionando {alias}...]\n"
                    continue

                elif kind == "on_tool_end":
                    active_tools -= 1
                    tool_name = event.get("name", "ferramenta")
                    alias = TOOL_ALIASES.get(tool_name, tool_name)
                    if active_tools == 0:
                        yield f"\n[✅ {alias} finalizado]\n"

                    if tool_name == "analyze_and_execute_sql":
                        sql_used = True
                    elif tool_name == "fetch_clinical_guidelines":
                        rag_used = True
                    continue

                elif kind == "on_chat_model_stream":
                    if active_tools == 0:
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            token = extract_text_from_content(chunk.content)
                            final_answer += token
                            yield token

        except asyncio.CancelledError:
            logger.warning("Streaming cancelado pelo cliente.")
            return
    else:
        response = await react_agent.ainvoke({"messages": messages})
        last_message = response["messages"][-1]
        final_answer = extract_text_from_content(last_message.content)
        
        for msg in response["messages"]:
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["name"] == "analyze_and_execute_sql":
                        sql_used = True
                    elif tc["name"] == "fetch_clinical_guidelines":
                        rag_used = True
        yield final_answer

    raw_athena = athena_results_context.get([])
    raw_rag = rag_results_context.get([])

    synth_result = {
        "final_answer": final_answer,
        "analysis_type": "react_agent_execution",
        "error": False,
        "rag_used": rag_used,
        "sql_used": sql_used,
        "job_id": job_id,
        "sessionId": effective_session,
        "conversationId": effective_conversation,
        "originalInput": message,
        "has_data": sql_used and len(raw_athena) > 0,
    }

    await add_user_message(effective_session, message)
    await add_ai_message(effective_session, final_answer)

    asyncio.create_task(_post_execution(
        job_id, effective_session, effective_conversation,
        message, synth_result, raw_athena, raw_rag
    ))


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

        lessons = generate_lessons_from_execution(result)
        if lessons:
            await save_learned_lessons(lessons)

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
        # Indexação on-demand: aproveita os IDs já retornados pelo Athena
        # para vetorizar em background, sem custo adicional de SQL.
        if raw_athena:
            ids = [
                r.get("id_atendimento")
                for entry in raw_athena
                for r in (entry.get("results") or [])
                if r.get("id_atendimento")
            ]
            if ids:
                asyncio.create_task(index_batch_by_ids(ids))
                logger.info(f"Indexação on-demand agendada para {len(ids)} atendimento(s).")
        flush_langsmith()
