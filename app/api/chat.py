from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from typing import Optional

from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key
from app.core.logger import logger

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
    api_key: str = Security(get_api_key),
):
    """
    Endpoint legado. Use /api/v1/iris/chat para session_id e cache.
    Mantido para compatibilidade com clientes existentes.
    """
    logger.info(f"Legacy /chat chamado | user_id: {req.user_id} | stream: {req.stream}")

    try:
        full_response = ""
        async for chunk in run_iris_agent(req.user_id, req.message, stream=False):
            if chunk:
                full_response += chunk

        if not full_response:
            return ChatResponse(response="", status="error", error="Nenhuma resposta foi gerada.")

        return ChatResponse(response=full_response, status="success", error=None)

    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e))
