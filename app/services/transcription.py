from openai import OpenAI
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str) -> str:
    """
    Transcreve um arquivo de áudio usando o modelo Whisper da OpenAI.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {file_path}")

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
        
        return transcript
    except Exception as e:
        logger.error(f"Erro na transcrição Whisper: {e}")
        raise e
