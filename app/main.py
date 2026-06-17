from dotenv import load_dotenv

# CRÍTICO: load_dotenv DEVE ser a primeira instrução antes de qualquer import
# do projeto, para garantir que as variáveis de ambiente estejam disponíveis
# quando os módulos forem carregados.
load_dotenv(override=True)

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.clients import aclose_clients
from app.core.logger import logger
from app.core.observability import configure_langsmith, flush_langsmith
from app.services.memory import close_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    - Startup: configura LangSmith.
    - Shutdown: faz flush de traces, fecha o pool do Postgres e os clientes HTTP
      compartilhados (httpx) para evitar conexões pendentes em redeploys.
    """
    # --- Startup ---
    configure_langsmith()
    logger.info("LangSmith configurado no startup.")

    yield

    # --- Shutdown ---
    logger.info("Encerrando aplicação...")
    try:
        flush_langsmith()
    except Exception as e:
        logger.warning(f"Falha no flush_langsmith: {e}")
    try:
        await close_pool()
    except Exception as e:
        logger.warning(f"Falha no close_pool: {e}")
    try:
        await aclose_clients()
    except Exception as e:
        logger.warning(f"Falha no aclose_clients: {e}")
    logger.info("Shutdown concluído.")


app = FastAPI(
    title="Iris AI Agent",
    version="1.0.0",
    description="Agente de inteligência clínica especializado em cirurgias de catarata.",
    lifespan=lifespan,
)

from app.core.config import settings
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.api.iris_chat import router as iris_router
from app.api.indexer_router import router as indexer_router

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
app.include_router(iris_router)
app.include_router(indexer_router)


@app.get("/")
def home():
    """Health check endpoint for Render monitoring."""
    logger.info("Health check endpoint called.")
    return {
        "status": "ok",
        "agent": "Iris",
        "version": "1.0.0",
        "environment": "production" if os.getenv("RENDER") else "development",
    }
