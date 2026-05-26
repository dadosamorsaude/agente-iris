from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Security
from app.agent.iris_orchestrator import run_iris_agent
from app.api.security import get_api_key
from app.services.transcription import transcribe_audio
from app.core.logger import logger
import os
import uuid
import shutil

router = APIRouter(prefix="/chat", tags=["voice"])

UPLOAD_DIR = "temp_audios"

@router.post("/voice")
async def chat_voice(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key),
):
    """
    Endpoint unificado para Lovable:
    1. Recebe áudio.
    2. Transcreve usando Whisper.
    3. Processa o texto com a Iris.
    4. Retorna a resposta final.
    """
    logger.info(f"Recebido pedido de chat por voz | user_id: {user_id}")

    # 1. Salva arquivo temporário
    ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
    temp_filename = f"voice_{uuid.uuid4()}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Transcreve
        logger.info(f"Transcrevendo áudio temporário: {temp_filename}")
        transcribed_text = transcribe_audio(temp_path)
        
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Não foi possível transcrever o áudio.")

        logger.info(f"Transcrição concluída: {transcribed_text[:50]}...")

        # 3. Executa o Agente com o texto transcrito + Instrução de Estruturação e Auditoria
        full_query = (
            "Com base na transcrição abaixo, realize as seguintes tarefas:\n"
            "1. Estruture o texto nos campos: ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA e CID-10.\n"
            "2. Realize uma auditoria de conformidade clínica baseada nas normas do CFM e RDCs, "
            "verificando se os campos clínicos atendem às réguas de qualidade da Iris.\n\n"
            f"Transcrição:\n{transcribed_text}"
        )
        
        full_response = ""
        async for chunk in run_iris_agent(user_id, full_query, stream=False):
            if chunk:
                full_response += chunk

        # 4. Limpa o arquivo temporário
        os.remove(temp_path)

        return {
            "status": "success",
            "transcription": transcribed_text,
            "response": full_response
        }

    except Exception as e:
        logger.error(f"Erro no processamento de voz: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
