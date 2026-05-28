from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import json

from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key

router = APIRouter(prefix="/api/v1/iris", tags=["Iris Agent"])


class IrisChatRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário para histórico/sessão")
    message: str = Field(alias="chatInput", description="Mensagem ou pergunta do usuário")
    stream: bool = Field(default=False, description="Habilitar streaming de resposta (SSE)")
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


@router.post("/chat")
async def chat_endpoint(req: IrisChatRequest, api_key: str = Depends(get_api_key)):
    """
    Endpoint principal para conversar com o Iris Agent (Deep Agent).
    Pode retornar uma resposta JSON consolidada ou um StreamingResponse (SSE).
    """

    if req.stream:
        async def event_generator():
            try:
                async for chunk in run_iris_agent(
                    user_id=req.user_id,
                    message=req.message,
                    stream=True,
                    session_id=req.session_id,
                    conversation_id=req.conversation_id,
                ):
                    # Formato SSE (Server-Sent Events)
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            except Exception:
                yield "data: {\"content\": \"Erro interno no streaming.\"}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                # Evita buffering em proxies (Nginx, Render, Cloudflare)
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        # Modo síncrono: agrega os chunks e retorna resposta única
        chunks = []
        async for chunk in run_iris_agent(
            user_id=req.user_id,
            message=req.message,
            stream=False,
            session_id=req.session_id,
            conversation_id=req.conversation_id,
        ):
            chunks.append(chunk)

        final_text = "".join(chunks)
        return {"response": final_text}
