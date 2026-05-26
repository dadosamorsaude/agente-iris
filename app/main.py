from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.api.audio import router as audio_router
from app.api.voice import router as voice_router
from app.api.ws import router as ws_router
from app.api.iris_chat import router as iris_router
from app.core.logger import logger
import time
import os
from dotenv import load_dotenv

# Carrega arquivos .env pro os.environ (essencial pro LangSmith enxergar as chaves no ambiente)
load_dotenv()

app = FastAPI(
    title="Iris AI Agent",
    version="1.0.0",
    description="Agente de inteligência clínica especializado em cirurgias de catarata.",
)

from app.core.config import settings

# Parse de domínios permitidos via variável de ambiente (separados por vírgula)
allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Completed {request.method} {request.url.path} with status {response.status_code} in {process_time:.3f}s")
    
    return response

app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(audio_router)
app.include_router(voice_router)
app.include_router(ws_router)
app.include_router(iris_router)


@app.get("/")
def home():
    """Health check endpoint for Render monitoring."""
    logger.info("Health check endpoint called.")
    return {
        "status": "ok",
        "agent": "Iris",
        "version": "1.0.0",
        "environment": "production" if os.getenv("RENDER") else "development"
    }