import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key
from app.core.logger import logger
from app.services.cache import semantic_cache
from app.tools.athena import athena_results_context
from app.tools.rag import rag_results_context

router = APIRouter(prefix="/api/v1/iris", tags=["Iris Agent"])


class IrisChatRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário para histórico/sessão")
    message: str = Field(alias="chatInput", description="Mensagem ou pergunta do usuário")
    stream: bool = Field(default=False, description="Habilitar streaming de resposta (SSE)")
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


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
            return {"response": text}

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

        return {"response": full_text}

    except Exception as e:
        logger.exception("Error in /api/v1/iris/chat")
        return {"response": "", "error": str(e)}
