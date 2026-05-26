"""
Agente Avaliador de Acurácia — LLM-as-Judge (Iris)

Compara a resposta da Iris com os dados brutos retornados pelo Athena,
gerando um score e justificativa detalhada dos erros encontrados de acordo com
os critérios estritos de qualidade do Judge do N8N (judge.json).
"""

import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt do Avaliador (Judge Oficial do N8N)
# ──────────────────────────────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """
Você é o Judge de Qualidade do Iris Catarata.

Função: auditar se a resposta candidata é 100% fiel aos dados brutos quantitativos disponíveis do Athena.

Você NÃO reescreve a resposta do agente. Apenas avalia sua acurácia e integridade.

BLOQUEIE a resposta (definindo should_block_callback=true) somente se houver falha CRÍTICA:
1. Inventou métricas, totais ou percentuais que não estão explicitamente contidos no resumo (summary) dos dados brutos.
2. Inventou registros, IDs de atendimento, datas ou evidências clínicas que não existem nos dados brutos.
3. Contradiz diretamente row_count ou summary das informações fornecidas.
4. Afirma que há dados disponíveis quando o resumo de dados está vazio (ou vice-versa).
5. Expõe a query SQL bruta na resposta quando o usuário NÃO pediu explicitamente pelo SQL (user_asked_for_sql=false).
6. Vaza termos de arquitetura interna da IA ou do fluxo (palavras proibidas: node, workflow, tool, payload, schema, judge, retry, output_mode, wants_rows).
7. Diz que a análise foi um "sucesso" ou correta mas houve algum erro (error=true).

Falhas menores (texto levemente truncado ou leve ambiguidade de redação) NÃO devem bloquear.

Retorne SOMENTE JSON válido (sem markdown, sem blocos ```json):
{
  "judge_passed": true,
  "overall_score": 0.95,
  "should_block_callback": false,
  "block_reason": null,
  "issues": []
}

Regras para Pontuação (overall_score):
- 0.90-1.00: excelente, dados perfeitamente fiéis.
- 0.75-0.89: aceitável, sem falhas críticas.
- 0.50-0.74: fraco, mas passa se não houver nenhuma falha crítica de fidelidade.
- <0.50: bloqueie obrigatoriamente.

Opções de block_reason possíveis se bloqueado:
- hallucinated_data
- contradiction_with_data
- architecture_leak
- sql_exposed_without_request
- success_but_error

No campo "issues", detalhe cada erro específico encontrado na resposta do agente.
"""

EVALUATOR_USER_TEMPLATE = """
## Pergunta do Usuário:
{user_question}

## Resposta Candidata da Iris:
{agent_response}

## Dados Brutos Disponíveis (Athena/SQL Result):
- SQL executada: {sql}
- Dados brutos: {raw_data}

## Contexto Clínico RAG (Régua de Catarata):
{rag_context}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Avaliador Principal
# ──────────────────────────────────────────────────────────────────────────────

def _get_evaluator_llm() -> ChatOpenAI:
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
    Avalia a acurácia da resposta da Iris e se ela obedece aos critérios
    de qualidade do Judge N8N, detectando alucinações de dados e vazamentos.

    Returns:
        dict: Payload estruturado conforme o Judge do n8n.
    """
    if not raw_athena_data and not rag_context:
        logger.warning("Avaliador: nenhum dado bruto do Athena nem RAG disponível.")
        return _empty_evaluation("Nenhum dado do Athena ou RAG disponível para avaliar.")

    # Formata dados do Athena
    raw_data_str = json.dumps(raw_athena_data or [], ensure_ascii=False, indent=2, default=str)
    
    # Obtém a última SQL executada se houver
    last_sql = ""
    if raw_athena_data and isinstance(raw_athena_data, list):
        last_sql = raw_athena_data[-1].get("sql", "")

    # Formata contexto do RAG
    if rag_context:
        rag_parts = []
        for item in rag_context:
            source = item.get("source", "Desconhecido")
            query = item.get("query", "")
            chunks = "\n---\n".join(item.get("chunks", []))
            rag_parts.append(f"[{source}] Query: '{query}'\n{chunks}")
        rag_context_str = "\n\n".join(rag_parts)
    else:
        rag_context_str = "Nenhuma diretriz clínica foi consultada."

    user_message = EVALUATOR_USER_TEMPLATE.format(
        user_question=user_question,
        agent_response=agent_response,
        sql=last_sql,
        raw_data=raw_data_str,
        rag_context=rag_context_str,
    )

    try:
        llm = _get_evaluator_llm()
        response = await llm.ainvoke([
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])

        raw_content = response.content.strip()

        # Remove markdown wrappers
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
            raw_content = raw_content.rsplit("```", 1)[0] if "```" in raw_content else raw_content

        parsed = json.loads(raw_content.strip())
        
        # Garante normalizações de score e aprovação idênticas ao processamento do N8N
        score = parsed.get("overall_score", 0.0)
        should_block = parsed.get("should_block_callback", False)
        
        # Aprovado se não bloqueado E score >= 0.75
        judge_passed = parsed.get("judge_passed", True) and not should_block and score >= 0.75

        evaluation = {
            "judge_passed": judge_passed,
            "overall_score": score,
            "should_block_callback": should_block,
            "block_reason": parsed.get("block_reason") if should_block else None,
            "issues": parsed.get("issues", []),
            "justificativa": parsed.get("justificativa", "Avaliação de auditoria clínica concluída."),
            "evaluated_at": datetime.utcnow().isoformat(),
            "model": settings.MODEL_NAME,
            "had_rag_context": bool(rag_context),
            "had_athena_data": bool(raw_athena_data)
        }

        logger.info(
            f"Avaliação concluída | score={evaluation.get('overall_score')} "
            f"| aprovado={evaluation.get('judge_passed')}"
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
        "judge_passed": False,
        "overall_score": 0.0,
        "should_block_callback": True,
        "block_reason": "judge_parse_error",
        "issues": [{"tipo": "judge_parse_error", "severidade": "alto", "message": reason}],
        "justificativa": reason,
        "evaluated_at": datetime.utcnow().isoformat(),
        "model": settings.MODEL_NAME,
    }

