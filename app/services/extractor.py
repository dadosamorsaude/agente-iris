import logging
import re
from typing import Optional
from pydantic import BaseModel, Field
from app.services.llm import get_chat_model_openai

logger = logging.getLogger(__name__)

class AnalysisResultSchema(BaseModel):
    classificacao: str = Field(description="Classificação do caso: 'positivo', 'provavel', 'negativo' ou 'pos_operatorio'")
    cpf: Optional[str] = Field(None, description="CPF do paciente identificado no prontuário/texto (ex: '123.456.789-00' ou '12345678900'). Retorne null se não encontrado.")
    termo_gatilho: Optional[str] = Field(None, description="Termo ou procedimento clínico que disparou a classificação, ex: 'facectomia OD'")
    evidencia_textual: Optional[str] = Field(None, description="Trecho exato e literal do prontuário que serve de evidência clínica")
    campo_origem: Optional[str] = Field(None, description="Campo de origem no prontuário, ex: 'conduta', 'anamnese', 'solicitacao', 'prescricao'")
    lateralidade: Optional[str] = Field(None, description="Lateralidade identificada: 'OD', 'OE', 'AO' ou null")
    tipo_termo: Optional[str] = Field(None, description="Tipo de termo clínico: 'procedimento' ou 'diagnostico'")
    verbo_acao: Optional[str] = Field(None, description="Verbo de ação associado: 'indicar', 'prescrever', 'solicitar' ou null")
    contexto: Optional[str] = Field(None, description="Contexto clínico resumido do caso")
    confianca: int = Field(..., description="Confiança na classificação de 0 a 100")


def extract_cpf_regex(text: str) -> Optional[str]:
    if not text:
        return None
    # 1. Procura por CPF formatado clássico: 000.000.000-00
    match = re.search(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', text)
    if match:
        return match.group(0)
    # 2. Procura por 11 dígitos numéricos isolados
    match = re.search(r'\b\d{11}\b', text)
    if match:
        return match.group(0)
    return None


async def extract_clinical_analysis(user_message: str, agent_response: str) -> Optional[dict]:
    """
    Analisa o prontuário clínico (user_message) e a resposta da Iris (agent_response)
    para extrair dados estruturados da auditoria clínica de catarata.
    """
    logger.info("Iniciando extração estruturada do caso clínico para o final do SSE...")
    try:
        llm = get_chat_model_openai(temperature=0.0)
        structured_llm = llm.with_structured_output(AnalysisResultSchema)

        prompt = (
            f"Você é um assistente especialista em estruturação de dados clínicos.\n"
            f"Sua tarefa é extrair um JSON com informações estruturadas da análise clínica a partir do prontuário "
            f"fornecido pelo usuário e do diagnóstico/classificação produzido pelo agente clínico.\n\n"
            f"### Mensagem do Usuário (Prontuário/Trecho):\n{user_message}\n\n"
            f"### Resposta da Iris (Agente):\n{agent_response}\n\n"
            f"Consolide e estruture os dados clínicos."
        )

        extracted: AnalysisResultSchema = await structured_llm.ainvoke(prompt)
        
        result_dict = extracted.model_dump()

        # Adiciona verificação robusta de CPF via regex se o LLM não capturou
        if not result_dict.get("cpf"):
            cpf_regex = extract_cpf_regex(user_message)
            if cpf_regex:
                result_dict["cpf"] = cpf_regex
                logger.info(f"CPF extraído via regex: {cpf_regex}")

        logger.info(f"Extração clínica concluída: {result_dict}")
        return result_dict

    except Exception as e:
        logger.error(f"Erro ao extrair análise estruturada clínica: {e}")
        # Fallback básico com extração de CPF via regex
        cpf_fallback = extract_cpf_regex(user_message)
        return {
            "classificacao": "negativo",
            "cpf": cpf_fallback,
            "termo_gatilho": None,
            "evidencia_textual": None,
            "campo_origem": None,
            "lateralidade": None,
            "tipo_termo": None,
            "verbo_acao": None,
            "contexto": None,
            "confianca": 50
        }
