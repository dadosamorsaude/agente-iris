from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.logger import logger
import os
import uuid

router = APIRouter(prefix="/audio", tags=["audio"])

UPLOAD_DIR = "temp_audios"

# Garante que a pasta exista
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio e salva temporariamente para transcrição.
    """
    # Validação simples de extensão
    allowed_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Extensão não permitida: {ext}")

    # Gera um nome único para o arquivo
    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Áudio recebido e salvo: {unique_filename}")
        
        return {
            "status": "success",
            "filename": unique_filename,
            "message": "Áudio recebido. Agora você pode pedir à Iris para analisá-lo."
        }
    except Exception as e:
        logger.error(f"Erro ao salvar áudio: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao salvar o arquivo.")
