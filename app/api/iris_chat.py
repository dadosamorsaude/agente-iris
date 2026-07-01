import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key
from app.core.logger import logger
from app.services.cache import semantic_cache
from app.services.mcp_client import athena_results_context, rag_results_context

router = APIRouter(prefix="/api/v1/iris", tags=["Iris Agent"])


class IrisChatRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário para histórico/sessão")
    message: str = Field(alias="chatInput", description="Mensagem ou pergunta do usuário")
    stream: bool = Field(default=False, description="Habilitar streaming de resposta (SSE)")
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class ReportRequest(BaseModel):
    period: str = Field(default="mes_atual", description="Período do relatório")
    stratification: str = Field(default="todos", description="Filtro de classificação/estratificação")
    laterality: str = Field(default="todos", description="Filtro de lateralidade")
    start_date: Optional[str] = Field(default=None, description="Data de início (opcional)")
    end_date: Optional[str] = Field(default=None, description="Data de fim (opcional)")
    clinic: Optional[str] = Field(default=None, description="Filtro de clínica (opcional)")


@router.post("/chat")
async def chat_endpoint(
    req: IrisChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
):
    """Endpoint principal do Iris Agent. Suporta cache semântico, histórico por sessão e SSE."""

    cached_response = await semantic_cache.get(req.message)

    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            if cached_response:
                text = cached_response.get("response")
                yield f"data: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
                
                # Executa extração estruturada de CPF/Caso Clínico para o cache
                try:
                    from app.services.extractor import extract_clinical_analysis
                    result_data = await extract_clinical_analysis(req.message, text)
                    if result_data:
                        yield f"data: {json.dumps({'result': result_data}, ensure_ascii=False)}\n\n"
                except Exception as ex:
                    logger.error(f"Erro ao extrair dados do cache: {ex}")

                yield "data: [DONE]\n\n"
                return

            try:
                full_stream_response = ""
                async for chunk in run_iris_agent(
                    user_id=req.user_id,
                    message=req.message,
                    stream=True,
                    session_id=req.session_id,
                    conversation_id=req.conversation_id,
                ):
                    if not chunk:
                        continue
                    full_stream_response += chunk
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                # No final do stream, executa a extração estruturada (CPF, classificação, etc)
                if full_stream_response:
                    try:
                        from app.services.extractor import extract_clinical_analysis
                        result_data = await extract_clinical_analysis(req.message, full_stream_response)
                        if result_data:
                            yield f"data: {json.dumps({'result': result_data}, ensure_ascii=False)}\n\n"
                    except Exception as ex:
                        logger.error(f"Erro ao extrair análise estruturada clínica no stream: {ex}")

                yield "data: [DONE]\n\n"

                if full_stream_response:
                    raw_athena = athena_results_context.get([])
                    raw_rag = rag_results_context.get([])
                    background_tasks.add_task(
                        semantic_cache.set,
                        req.message, full_stream_response, raw_athena, raw_rag
                    )
            except Exception as e:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    try:
        if cached_response:
            text = cached_response.get("response")
            # Extrai o result
            from app.services.extractor import extract_clinical_analysis
            result_data = await extract_clinical_analysis(req.message, text)
            return {"response": text, "result": result_data}

        chunks = []
        async for chunk in run_iris_agent(
            user_id=req.user_id,
            message=req.message,
            stream=False,
            session_id=req.session_id,
            conversation_id=req.conversation_id,
        ):
            chunks.append(chunk)

        full_text = "".join(chunks)

        raw_athena = athena_results_context.get([])
        raw_rag = rag_results_context.get([])
        background_tasks.add_task(
            semantic_cache.set,
            req.message, full_text, raw_athena, raw_rag
        )

        from app.services.extractor import extract_clinical_analysis
        result_data = await extract_clinical_analysis(req.message, full_text)
        return {"response": full_text, "result": result_data}

    except Exception as e:
        logger.exception("Error in /api/v1/iris/chat")
        return {"response": "", "error": str(e)}


@router.post("/report")
async def report_endpoint(
    req: ReportRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Endpoint de relatório estruturado para análise de cirurgias de catarata.
    Retorna os dados dos pacientes com base nos filtros selecionados.
    """
    import datetime
    from fastapi import HTTPException
    
    period = req.period
    stratification = req.stratification
    laterality = req.laterality
    start_date = req.start_date
    end_date = req.end_date
    clinic = req.clinic
    
    logger.info(f"Requisição POST /report | period={period} | stratification={stratification} | laterality={laterality} | start_date={start_date} | end_date={end_date} | clinic={clinic}")
    
    if clinic and len(clinic) > 200:
        raise HTTPException(status_code=400, detail="O nome da clínica deve ter no máximo 200 caracteres.")
    
    # 1. Determina as datas do período
    def calculate_dates(p: str) -> tuple[Optional[str], Optional[str]]:
        today = datetime.date.today()
        if p == "hoje":
            return today.isoformat(), today.isoformat()
        elif p == "ontem":
            yesterday = today - datetime.timedelta(days=1)
            return yesterday.isoformat(), yesterday.isoformat()
        elif p == "ultimos_7_dias":
            start = today - datetime.timedelta(days=6)
            return start.isoformat(), today.isoformat()
        elif p == "ultimos_30_dias":
            start = today - datetime.timedelta(days=29)
            return start.isoformat(), today.isoformat()
        elif p == "mes_atual":
            start = today.replace(day=1)
            return start.isoformat(), today.isoformat()
        elif p == "mes_passado":
            first_this = today.replace(day=1)
            last_prev = first_this - datetime.timedelta(days=1)
            first_prev = last_prev.replace(day=1)
            return first_prev.isoformat(), last_prev.isoformat()
        elif p == "todo_historico":
            return None, None
        return None, None

    if start_date or end_date:
        start_date_str = start_date
        end_date_str = end_date
    else:
        start_date_str, end_date_str = calculate_dates(period)

    # 2. Constrói o texto do período
    if start_date_str and end_date_str:
        period_text = f"de {start_date_str} ate {end_date_str}"
    elif start_date_str:
        period_text = f"a partir de {start_date_str}"
    elif end_date_str:
        period_text = f"ate {end_date_str}"
    else:
        period_text = "de todo o historico sem filtro de periodo"

    # 3. Constrói a query para o SQL analyst
    query_text = f"caracterizacao dos pacientes por classificacao clinica {period_text}"

    try:
        # 4. Obtém o contexto clínico (RAG)
        from app.agent.specialists.clinical_rag import clinical_rag_expert
        rag_json_str = await clinical_rag_expert("diretrizes de catarata")
        try:
            rag_data = json.loads(rag_json_str)
            rag_context = rag_data.get("rag_context", rag_json_str)
        except Exception:
            rag_context = rag_json_str

        # 5. Executa a análise SQL Athena
        from app.agent.specialists.sql_analyst import sql_analyst_expert
        result = await sql_analyst_expert(
            query=query_text,
            rag_context=rag_context,
            hoje=datetime.date.today().isoformat()
        )

        if result.get("execution_status") == "error":
            logger.error(f"Erro ao executar relatório no Athena: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", {}).get("message", "Erro desconhecido na consulta SQL"),
                "summary": {},
                "filtered_summary": {},
                "rows": [],
                "sql": result.get("sql"),
                "limitations": result.get("limitations", [])
            }

        all_rows = result.get("rows", [])
        
        # Deduplica as linhas por id_atendimento para garantir registros únicos por atendimento
        seen = set()
        unique_rows = []
        for r in all_rows:
            key = r.get("id_atendimento") or r.get("id_paciente") or json.dumps(r, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_rows.append(r)
        all_rows = unique_rows
        
        # 6. Calcula o resumo geral (overall summary) do período
        # Classificações possíveis: positivos, provaveis, negativos, pos_operatorios
        def compute_summary_stats(rows_list: list[dict]) -> dict:
            total = len(rows_list)
            positivos = sum(1 for r in rows_list if r.get("classificacao") == "positivo")
            provaveis = sum(1 for r in rows_list if r.get("classificacao") == "provavel")
            negativos = sum(1 for r in rows_list if r.get("classificacao") == "negativo")
            pos_op = sum(1 for r in rows_list if r.get("classificacao") == "pos_operatorio")
            
            # Contagem de pacientes únicos
            pacientes_unicos = len({r.get("id_paciente") for r in rows_list if r.get("id_paciente") is not None})
            
            return {
                "total_registros": total,
                "total_pacientes_unicos": pacientes_unicos,
                "positivos": positivos,
                "provaveis": provaveis,
                "negativos": negativos,
                "pos_operatorios": pos_op,
                "percentual_positivos": round(positivos / total * 100, 2) if total > 0 else 0,
                "percentual_provaveis": round(provaveis / total * 100, 2) if total > 0 else 0,
                "percentual_negativos": round(negativos / total * 100, 2) if total > 0 else 0,
                "percentual_pos_operatorios": round(pos_op / total * 100, 2) if total > 0 else 0
            }

        overall_summary = compute_summary_stats(all_rows)

        # 7. Aplica os filtros in-memory
        filtered_rows = all_rows

        # Filtro de classificação / estratificação
        # Valores permitidos no frontend: todos, positivo, provavel, negativo, pos_operatorio
        filter_strat = stratification.lower().strip()
        if filter_strat != "todos":
            filtered_rows = [r for r in filtered_rows if r.get("classificacao") == filter_strat]

        # Filtro de lateralidade
        # Valores permitidos no frontend: todos, OD, OE, AO, null
        filter_lat = laterality.upper().strip()
        if filter_lat != "TODOS":
            if filter_lat == "NULL":
                filtered_rows = [r for r in filtered_rows if r.get("lateralidade") is None]
            else:
                filtered_rows = [r for r in filtered_rows if r.get("lateralidade") == filter_lat]

        # Filtro de clínica (case-insensitive substring match)
        if clinic:
            clinic_normalized = clinic.lower().strip()
            filtered_rows = [
                r for r in filtered_rows
                if r.get("clinica") and clinic_normalized in str(r.get("clinica")).lower()
            ]

        # 8. Calcula o resumo para as linhas filtradas
        filtered_summary = compute_summary_stats(filtered_rows)

        return {
            "success": True,
            "summary": overall_summary,
            "filtered_summary": filtered_summary,
            "rows": filtered_rows,
            "sql": result.get("sql"),
            "limitations": result.get("limitations", []),
            "filters_applied": {
                "period": period,
                "stratification": stratification,
                "laterality": laterality,
                "start_date": start_date,
                "end_date": end_date,
                "clinic": clinic
            }
        }

    except Exception as e:
        logger.exception("Erro ao processar relatório")
        return {
            "success": False,
            "error": str(e),
            "summary": {},
            "filtered_summary": {},
            "rows": [],
            "sql": None,
            "limitations": [str(e)],
            "filters_applied": {
                "period": period,
                "stratification": stratification,
                "laterality": laterality,
                "start_date": start_date,
                "end_date": end_date,
                "clinic": clinic
            }
        }


# Cache global para a lista de clínicas
_clinics_cache = {
    "data": None,
    "timestamp": 0
}
CLINICS_CACHE_TTL = 300  # 5 minutos

@router.get("/clinics")
async def clinics_endpoint(
    api_key: str = Depends(get_api_key),
):
    """
    Retorna a lista distinta de clínicas disponíveis na base, ordenada alfabeticamente (pt-BR).
    Utiliza um cache em memória de 5 minutos.
    """
    import time
    import unicodedata
    import asyncio
    
    now = time.time()
    if _clinics_cache["data"] is not None and now - _clinics_cache["timestamp"] < CLINICS_CACHE_TTL:
        logger.info("Retornando lista de clínicas do cache.")
        return {"success": True, "clinics": _clinics_cache["data"]}

    logger.info("Buscando lista de clínicas no Athena...")
    sql = "SELECT DISTINCT clinica FROM pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia WHERE clinica IS NOT NULL AND clinica <> ''"
    
    try:
        from app.services.mcp_client import query_athena_tool
        import json
        results_str = await query_athena_tool._arun(sql)
        results = json.loads(results_str)
        
        raw_clinics = [row["clinica"] for row in results if "clinica" in row and row["clinica"]]
        
        # Ordenação pt-BR (ignora acentos ao ordenar de forma case-insensitive)
        def pt_br_sort_key(s: str) -> str:
            normalized = unicodedata.normalize('NFD', s)
            return "".join(c for c in normalized if unicodedata.category(c) != 'Mn').lower()
            
        sorted_clinics = sorted(list(set(raw_clinics)), key=pt_br_sort_key)
        
        # Salva no cache
        _clinics_cache["data"] = sorted_clinics
        _clinics_cache["timestamp"] = now
        
        return {"success": True, "clinics": sorted_clinics}
        
    except Exception as e:
        logger.exception("Erro ao buscar clínicas")
        return {"success": False, "clinics": [], "error": str(e)}

