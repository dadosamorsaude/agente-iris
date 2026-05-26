"""
Agente Avaliador de Acurácia — LLM-as-Judge (Iris)

Compara a resposta da Iris com os dados brutos retornados pelo Athena,
gerando um score de 0-100 com justificativa detalhada dos erros encontrados.
"""

import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt do Avaliador
# ──────────────────────────────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """
Você é um avaliador especializado em análise de prontuários médicos e dados clínicos.

Sua única função é avaliar se a RESPOSTA DO AGENTE reflete com precisão e completude
os DADOS BRUTOS do banco de dados E aplica corretamente as DIRETRIZES NORMATIVAS recuperadas.

## Critérios de Avaliação

1. **Precisão Factual (0–35 pts)**
   - Os números, contagens, percentuais e valores citados estão corretos?
   - Há distorções, arredondamentos indevidos ou dados inventados?

2. **Completude (0–25 pts)**
   - A resposta abordou os dados mais relevantes disponíveis?
   - Algum dado importante foi omitido sem justificativa?

3. **Interpretação Clínica (0–25 pts)**
   - A análise/conclusão está alinhada com o que os dados mostram?
   - O agente fez inferências incorretas ou generalizações indevidas?

4. **Aplicação Normativa (0–15 pts)**
   - O agente usou corretamente as diretrizes CFM/POPs recuperadas para embasar sua análise?
   - Ignorou normas relevantes que foram recuperadas? Aplicou normas em contexto equivocado?
   - Se não houver contexto normativo, atribua 15 automaticamente.

## Regras
- Avalie somente com base nos dados e diretrizes fornecidos, não em conhecimento prévio.
- Se os dados brutos estiverem vazios, retorne score 0 com justificativa.
- Seja objetivo e específico nos erros encontrados.
- Responda APENAS com o JSON solicitado, sem texto adicional.
"""

EVALUATOR_USER_TEMPLATE = """
## Dados Brutos do Athena:
{raw_data}

## Contexto Normativo Recuperado (CFM / POPs):
{rag_context}

## Pergunta do Usuário:
{user_question}

## Resposta da Iris:
{agent_response}

## Histórico da Conversa (Memória):
{chat_history}

## Avalie e responda APENAS em JSON:
{{
  "score": <inteiro 0-100>,
  "precisao_factual": <inteiro 0-35>,
  "completude": <inteiro 0-25>,
  "interpretacao_clinica": <inteiro 0-25>,
  "aplicacao_normativa": <inteiro 0-15>,
  "aprovado": <true se score >= 70, false caso contrário>,
  "erros_encontrados": [<lista de strings descrevendo cada erro específico>],
  "justificativa": "<resumo objetivo da avaliação em 2-3 frases>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Avaliador Principal
# ──────────────────────────────────────────────────────────────────────────────

def _get_evaluator_llm() -> ChatOpenAI:
    """Retorna o LLM avaliador (conforme MODEL_NAME configurado, temperatura 0 para consistência)."""
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY,
    )


async def evaluate_response(
    user_question: str,
    agent_response: str,
    raw_athena_data: list[dict],
    rag_context: list[dict] | None = None,
    chat_history: str = "",
) -> dict:
    """
    Avalia a acurácia da resposta da Iris considerando:
    - Dados brutos do Athena (dados quantitativos)
    - Contexto normativo do RAG (régua de catarata) usado na interpretação

    Args:
        user_question: A pergunta original do usuário.
        agent_response: A resposta gerada pela Iris.
        raw_athena_data: Lista de dicts com {sql, results} capturados durante a execução.
        rag_context: Lista de dicts com {source, query, chunks} do RAG (opcional).

    Returns:
        Dict com score, breakdown, erros e justificativa.
    """
    if not raw_athena_data and not rag_context:
        logger.warning("Avaliador: nenhum dado bruto do Athena nem RAG disponível.")
        return _empty_evaluation("Nenhum dado do Athena ou RAG disponível para avaliar.")

    # Formata dados do Athena
    raw_data_str = json.dumps(raw_athena_data or [], ensure_ascii=False, indent=2, default=str)

    # Formata contexto normativo do RAG
    if rag_context:
        rag_parts = []
        for item in rag_context:
            source = item.get("source", "Desconhecido")
            query = item.get("query", "")
            chunks = "\n---\n".join(item.get("chunks", []))
            rag_parts.append(f"[{source}] Query: '{query}'\n{chunks}")
        rag_context_str = "\n\n".join(rag_parts)
    else:
        rag_context_str = "Nenhuma diretriz normativa foi consultada nesta resposta."

    user_message = EVALUATOR_USER_TEMPLATE.format(
        raw_data=raw_data_str,
        rag_context=rag_context_str,
        user_question=user_question,
        agent_response=agent_response,
        chat_history=chat_history or "Nenhum histórico anterior.",
    )

    try:
        llm = _get_evaluator_llm()
        response = await llm.ainvoke([
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])

        raw_content = response.content.strip()

        # Remove possíveis blocos de código markdown (```json ... ```)
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]

        evaluation = json.loads(raw_content)
        evaluation["evaluated_at"] = datetime.utcnow().isoformat()
        evaluation["model"] = settings.MODEL_NAME
        evaluation["had_rag_context"] = bool(rag_context)
        evaluation["had_athena_data"] = bool(raw_athena_data)

        logger.info(
            f"Avaliação concluída | score={evaluation.get('score')} "
            f"| aprovado={evaluation.get('aprovado')}"
        )
        return evaluation

    except json.JSONDecodeError as e:
        logger.error(f"Avaliador: resposta do LLM não é JSON válido: {e}")
        return _empty_evaluation(f"Falha ao parsear resposta do avaliador: {e}")

    except Exception as e:
        logger.exception("Avaliador: erro ao invocar LLM avaliador")
        return _empty_evaluation(f"Erro interno no avaliador: {e}")


def _empty_evaluation(reason: str) -> dict:
    """Retorna uma avaliação vazia com score 0 quando não é possível avaliar."""
    return {
        "score": 0,
        "precisao_factual": 0,
        "completude": 0,
        "interpretacao_clinica": 0,
        "aprovado": False,
        "erros_encontrados": [reason],
        "justificativa": reason,
        "evaluated_at": datetime.utcnow().isoformat(),
        "model": settings.MODEL_NAME,
    }
