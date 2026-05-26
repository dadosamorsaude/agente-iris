from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_scheme)):
    if not settings.AGENTE_API_KEY:
        return True

    if api_key_header != settings.AGENTE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API Key",
        )

    return api_key_header