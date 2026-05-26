import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Security, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key
from app.core.logger import logger
from app.services.cache import semantic_cache
from app.tools.athena import athena_results_context
from app.tools.rag import rag_results_context

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    error: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Security(get_api_key),
):
    """
    Endpoint principal de chat.
    - stream=False: retorna JSON padrão
    - stream=True: retorna Server-Sent Events (SSE)
    """
    logger.info(f"Received chat request | user_id: {req.user_id} | stream: {req.stream}")

    # 1. Verifica no Cache Semântico
    cached_response = await semantic_cache.get(req.message)

    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            # Se deu HIT no cache, entrega e aciona evaluator
            if cached_response:
                text = cached_response.get("response")
                payload = {"text": text}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
                
            try:
                full_stream_response = ""
                async for chunk in run_iris_agent(req.user_id, req.message, stream=True):
                    if not chunk:
                        continue
                    
                    full_stream_response += chunk
                    payload = {"text": chunk}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

                if full_stream_response and not full_stream_response.startswith("Erro técnico:"):
                    # Captura dados brutos para o cache
                    raw_athena = athena_results_context.get([])
                    raw_rag = rag_results_context.get([])
                    background_tasks.add_task(
                        semantic_cache.set, 
                        req.message, full_stream_response, raw_athena, raw_rag
                    )

            except Exception as e:
                logger.exception("Streaming error")
                payload = {"error": str(e)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Tratamento quando stream=False
    try:
        # Se HIT no cache, entrega e aciona evaluator
        if cached_response:
            text = cached_response.get("response")
            return ChatResponse(response=text, status="success", error=None)

        full_response = ""

        async for chunk in run_iris_agent(req.user_id, req.message, stream=False):
            if chunk:
                full_response += chunk

        if not full_response:
            return ChatResponse(
                response="",
                status="error",
                error="Nenhuma resposta foi gerada.",
            )

        if full_response.startswith("Erro técnico:"):
            return ChatResponse(
                response="",
                status="error",
                error=full_response,
            )
            
        # Captura dados brutos para o cache
        raw_athena = athena_results_context.get([])
        raw_rag = rag_results_context.get([])
        background_tasks.add_task(
            semantic_cache.set, 
            req.message, full_response, raw_athena, raw_rag
        )

        return ChatResponse(
            response=full_response,
            status="success",
            error=None,
        )

    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e))