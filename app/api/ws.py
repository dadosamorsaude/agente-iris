from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.agent.iris_orchestrator import run_iris_agent
from app.services.transcription import transcribe_audio
from app.core.config import settings
from app.core.logger import logger
import os
import uuid
import json

router = APIRouter(tags=["voice-streaming"])

UPLOAD_DIR = "temp_audios"

@router.websocket("/ws/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket,
    api_key: str = Query(None)
):
    """
    WebSocket otimizado para Lovable/Deno Proxy.
    Protocolo:
    - Handshake: { "type": "start", "mime_type": "...", "sample_rate": ... }
    - Data: Binary chunks (WebM/Opus)
    - Finish: { "type": "stop" }
    """
    # 1. Validação de API Key via Query Param (Necessário para Deno Proxy)
    if settings.AGENTE_API_KEY and api_key != settings.AGENTE_API_KEY:
        await websocket.close(code=4003) # Forbidden
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"Conexão Voice-WS estabelecida | session_id: {session_id}")

    temp_filename = f"stream_{session_id}.webm"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    audio_file = None
    is_recording = False

    try:
        while True:
            message = await websocket.receive()
            
            # A) TRATAMENTO DE BINÁRIO (CHUNKS DE ÁUDIO)
            if "bytes" in message:
                if is_recording and audio_file:
                    audio_file.write(message["bytes"])
                continue

            # B) TRATAMENTO DE TEXTO (COMANDOS JSON)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "start":
                        logger.info(f"Iniciando gravação | session_id: {session_id}")
                        audio_file = open(temp_path, "wb")
                        is_recording = True
                        await websocket.send_json({"type": "partial", "text": "Gravando áudio..."})

                    elif msg_type == "stop":
                        if not is_recording or not audio_file:
                            continue
                        
                        logger.info(f"Finalizando gravação | session_id: {session_id}")
                        is_recording = False
                        audio_file.close()
                        audio_file = None

                        await websocket.send_json({"type": "partial", "text": "Processando transcrição..."})

                        # 1. Transcrição com Whisper
                        transcribed_text = transcribe_audio(temp_path)
                        
                        if not transcribed_text:
                            await websocket.send_json({"type": "error", "message": "Não foi possível transcrever o áudio."})
                            continue

                        # Envia o texto final da transcrição
                        await websocket.send_json({"type": "final", "text": transcribed_text})

                        # 2. Análise Clínica Automática (Iris)
                        await websocket.send_json({"type": "partial", "text": "Realizando análise de conformidade clínica..."})
                        
                        full_query = (
                            "Com base na transcrição abaixo, realize as seguintes tarefas:\n"
                            "1. Estruture o texto nos campos: ANAMNESE, CONDUTA, HIPÓTESE DIAGNÓSTICA e CID-10.\n"
                            "2. Realize uma auditoria de conformidade clínica baseada nas normas do CFM e RDCs, "
                            "verificando se os campos clínicos atendem às réguas de qualidade da Iris.\n\n"
                            f"Transcrição:\n{transcribed_text}"
                        )

                        full_response = ""
                        async for chunk in run_iris_agent(user_id=session_id, message=full_query, stream=True, session_id=session_id):
                            if chunk:
                                full_response += chunk
                                # Enviamos cada pedaço da análise para o front
                                await websocket.send_json({"type": "partial", "text": chunk})
                        
                        # Resposta final da análise
                        await websocket.send_json({
                            "type": "final", 
                            "text": full_response,
                            "is_analysis": True 
                        })

                except json.JSONDecodeError:
                    logger.warning(f"Recebido texto não-JSON no WS: {message['text']}")
                    continue

    except WebSocketDisconnect:
        logger.info(f"Voice-WS desconectado | session_id: {session_id}")
    except Exception as e:
        logger.error(f"Erro no Voice-WS: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if audio_file:
            audio_file.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
