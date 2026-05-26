from langchain_core.tools import tool
from langsmith import traceable
from app.services.transcription import transcribe_audio
import logging
import os

logger = logging.getLogger(__name__)

# Pasta temporária para áudios (deve bater com a configuração do main.py)
UPLOAD_DIR = "temp_audios"

@tool
@traceable(name="transcribe_audio_tool")
def transcribe_audio_tool(filename: str) -> str:
    """
    Transcreve um arquivo de áudio previamente enviado. 
    Use esta ferramenta quando o usuário enviar um áudio para análise.
    O 'filename' deve ser o nome do arquivo salvo no sistema.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        logger.info(f"Iniciando transcrição do arquivo: {file_path}")
        text = transcribe_audio(file_path)
        
        return f"Transcrição concluída com sucesso:\n\n{text}"
    except Exception as e:
        logger.error(f"Erro na ferramenta de transcrição: {e}")
        return f"Erro ao transcrever o áudio: {str(e)}"
