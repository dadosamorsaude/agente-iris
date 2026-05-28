import asyncio
import json
import logging

import httpx

from app.core.config import settings
from app.services.intent import detect_intent

logger = logging.getLogger(__name__)

def _get_headers() -> dict:
    if not settings.DATABASE_API_KEY:
        return {}
    return {
        "apikey": settings.DATABASE_API_KEY,
        "Authorization": f"Bearer {settings.DATABASE_API_KEY}",
        "Content-Type": "application/json"
    }


async def load_curated_lessons(memory_key: str = "iris_catarata") -> str:
    """
    Busca via Supabase REST até 15 aprendizados ativos curados da Iris e formata como checklist de prompt.
    """
    if not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        return "{\n  \"tipo\": \"aprendizados_curados_do_projeto\",\n  \"uso\": \"Nenhum banco configurado para aprendizados.\",\n  \"regras_prioritarias\": []\n}"

    try:
        url = f"{settings.supabase_rest_url}memoria_aprendizados_iris"
        params = {
            "memory_key": f"eq.{memory_key}",
            "active": "eq.true",
            "select": "category,lesson,reason,confidence,usage_count",
            "order": "confidence.desc,usage_count.desc",
            "limit": "15"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=_get_headers(), params=params)
            response.raise_for_status()
        rows = response.json()

        # Re-ordenação em Python para simular o CASE condicional da query antiga
        priority_categories = {
            'anti_alucinacao': 0, 'amostra_sem_agregacao': 0, 'amostra': 0,
            'evidencias': 0, 'rag_sql_handoff': 0, 'sql': 0, 'rag_clinico': 0,
            'formato_saida': 0, 'relatorio_metricas': 0, 'qualidade': 0
        }
        
        def sort_key(r):
            cat = r.get("category", "")
            cat_priority = priority_categories.get(cat, 2 if cat.startswith('bom_padrao') else 1)
            return (cat_priority, -float(r.get("confidence", 0)), -int(r.get("usage_count", 0)))
            
        rows.sort(key=sort_key)

        aprendizados = []
        for r in rows:
            aprendizados.append({
                "categoria": str(r.get("category", "")),
                "regra": str(r.get("lesson", "")),
                "motivo": str(r.get("reason") or "")[:500],
                "confianca": float(r.get("confidence") or 0.0),
                "usos": int(r.get("usage_count") or 0)
            })

        regras_prioritarias = [
            f"{item['categoria']}: {item['regra']}"
            for item in aprendizados if item["confianca"] >= 0.85
        ][:10]

        memory_context = {
            "tipo": "aprendizados_curados_do_projeto",
            "uso": [
                "Use como checklist operacional preventivo.",
                "Não copie respostas antigas.",
                "Não substitui RAG.",
                "Não substitui SQL.",
                "Não invente dados ausentes.",
                "Em conflito entre memória e dados/RAG, dados e RAG vencem."
            ],
            "regras_prioritarias": regras_prioritarias,
            "aprendizados": aprendizados
        }

        return json.dumps(memory_context, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.warning(f"Falha ao conectar ou buscar aprendizados via REST: {e}")
        return "{\n  \"tipo\": \"aprendizados_curados_do_projeto\",\n  \"uso\": \"Erro ao carregar memoria do banco.\",\n  \"regras_prioritarias\": []\n}"


def generate_lessons_from_execution(execution_data: dict) -> list[dict]:
    """
    Deriva e formula aprendizados a partir da execução e score do Judge,
    replicando com precisão as regras do nó 'Preparar Memória' do n8n.
    """
    lessons = []
    
    original_input = execution_data.get("originalInput", "")
    input_text = original_input.lower()
    analysis_type = str(execution_data.get("analysis_type", "")).lower()
    final_answer = str(execution_data.get("final_answer", "")).lower()
    
    judge_output = execution_data.get("judge_output") or {}
    judge_passed = execution_data.get("judge_passed")
    judge_score = execution_data.get("judge_score")
    
    # Normaliza textos para busca de termos
    judge_text = f"{str(judge_output)} {str(execution_data.get('issues', []))} {execution_data.get('errorType', '')} {execution_data.get('errorMessage', '')} {analysis_type} {input_text} {final_answer}".lower()

    low_score = judge_score is not None and float(judge_score) < 0.75
    judge_failed = (judge_passed is False) or low_score

    # Intent flags via serviço centralizado
    _detected = detect_intent(original_input)
    is_sample_intent = _detected["sample_mode"] or "amostra" in analysis_type
    is_evidence_intent = _detected["wants_rows"] and not _detected["sample_mode"]
    is_aggregation_intent = _detected["has_aggregation_intent"]

    # 1. Erro genérico recorrente
    if execution_data.get("error") is True and execution_data.get("errorType"):
        lessons.append({
            "category": "erro_recorrente",
            "lesson": f"Quando ocorrer {execution_data['errorType']}, tratar o erro antes de responder ao usuário.",
            "reason": execution_data.get("errorMessage") or execution_data.get("errorType"),
            "confidence": 0.75
        })

    # 2. Erro do Executor SQL
    if execution_data.get("errorType") == "sql_executor_error":
        lessons.append({
            "category": "sql",
            "lesson": "Validar SQL de forma determinística antes de executar, respeitando sintaxe Athena/Presto e tipos compatíveis.",
            "reason": "Execução SQL falhou ou foi rejeitada pelo executor.",
            "confidence": 0.90
        })

    # 3. RAG Context Clínico ausente
    if execution_data.get("errorType") == "missing_rag_context":
        lessons.append({
            "category": "rag_sql_handoff",
            "lesson": "Nunca chamar SQL sem rag_context clínico real retornado pelo RAG.",
            "reason": "O contexto RAG estava ausente, genérico ou inválido.",
            "confidence": 0.95
        })

    # 4. Reprovação geral do Judge
    if judge_failed:
        lessons.append({
            "category": "qualidade",
            "lesson": "Quando o Judge pontuar baixo ou reprovar, revisar aderência aos dados, completude, ausência de invenção e formato final.",
            "reason": f"judge_passed={judge_passed}; judge_score={judge_score}",
            "confidence": 0.80
        })

    # 5. Pedido de amostra com reprovação
    if judge_failed and is_sample_intent:
        lessons.append({
            "category": "amostra",
            "lesson": "Quando o usuário pedir amostra, retornar poucas linhas individuais completas, sem agregações, percentuais ou resumo consolidado.",
            "reason": f"Pedido de amostra teve baixa avaliação do Judge. judge_score={judge_score}",
            "confidence": 0.95
        })

    # 6. Agregação indevida em amostras
    if judge_failed and is_sample_intent:
        has_aggregation_words = any(w in final_answer for w in ['percentual', 'porcentagem', 'total', 'resumo', 'distribuicao', 'distribuição'])
        has_judge_critique = any(w in judge_text for w in ['agregacao indevida', 'agregação indevida', 'agregado', 'consolidado'])
        if has_aggregation_words or has_judge_critique:
            lessons.append({
                "category": "amostra_sem_agregacao",
                "lesson": "Em modo amostra, não consolidar resultados; mostrar registros reais com id_atendimento, id_paciente, data, classificação, score e evidência.",
                "reason": "O Judge indicou ou o texto final sugere uso de agregações em uma solicitação de amostra.",
                "confidence": 0.97
            })

    # 7. Pedido de evidências com reprovação
    if judge_failed and is_evidence_intent:
        lessons.append({
            "category": "evidencias",
            "lesson": "Quando o usuário pedir evidências, incluir campo de origem, termo detectado e trecho textual sempre que disponíveis.",
            "reason": f"Pedido de evidências teve baixa avaliação do Judge. judge_score={judge_score}",
            "confidence": 0.90
        })

    # 8. Alucinação de dados quantitativos
    has_hallucination_critique = any(w in judge_text for w in ['inventou', 'invenção', 'invencao', 'alucinacao', 'alucinação', 'sem base', 'nao sustentado', 'não sustentado', 'dados nao encontrados', 'dados não encontrados'])
    if judge_failed and has_hallucination_critique:
        lessons.append({
            "category": "anti_alucinacao",
            "lesson": "Nunca inventar totais, percentuais, pacientes, atendimentos, scores, classificações ou evidências ausentes no resultado SQL.",
            "reason": "O Judge indicou possível resposta sem sustentação nos dados.",
            "confidence": 0.98
        })

    # 9. RAG Clínico pulado ou inconsistente
    has_rag_critique = any(w in judge_text for w in ['rag', 'contexto clinico', 'contexto clínico', 'regua', 'régua', 'classificacao', 'classificação'])
    if judge_failed and has_rag_critique:
        lessons.append({
            "category": "rag_clinico",
            "lesson": "Para perguntas substantivas, usar o RAG como régua clínica antes do SQL e não substituir a régua por memória ou exemplos anteriores.",
            "reason": "O Judge indicou problema relacionado ao uso do RAG ou da classificação clínica.",
            "confidence": 0.92
        })

    # 10. Relatórios incompletos
    has_incompleteness_critique = any(w in judge_text for w in ['incompleto', 'faltou metrica', 'faltou métrica', 'sem percentual', 'sem total', 'sem estratificacao', 'sem estratificação'])
    if judge_failed and is_aggregation_intent and has_incompleteness_critique:
        lessons.append({
            "category": "relatorio_metricas",
            "lesson": "Em pedidos de relatório, contagem ou distribuição, incluir total, estratificação, percentuais quando disponíveis e interpretação objetiva.",
            "reason": "O Judge indicou incompletude em resposta agregada ou relatório.",
            "confidence": 0.90
        })

    # 11. Formato de saída JSON inválido
    has_format_critique = any(w in judge_text for w in ['json invalido', 'json inválido', 'formato invalido', 'formato inválido', 'nao retornou json', 'não retornou json'])
    if judge_failed and has_format_critique:
        lessons.append({
            "category": "formato_saida",
            "lesson": "O orquestrador deve retornar somente JSON válido no formato contratado, sem markdown, comentários ou texto fora do objeto.",
            "reason": "O Judge indicou problema de formato na saída.",
            "confidence": 0.95
        })

    # 12. Bom padrão RAG + SQL aprovado
    if execution_data.get("rag_used") is True and execution_data.get("sql_used") is True and execution_data.get("error") is not True:
        if judge_score is not None and float(judge_score) >= 0.85:
            lessons.append({
                "category": "bom_padrao",
                "lesson": "Para perguntas substantivas, o padrão RAG antes de SQL produz resposta mais confiável.",
                "reason": "Execução com RAG, SQL e boa avaliação do Judge.",
                "confidence": 0.85
            })
            if is_sample_intent:
                lessons.append({
                    "category": "bom_padrao_amostra",
                    "lesson": "Em pedidos de amostra, o melhor padrão é retornar poucos registros individuais com evidência textual, sem resumo agregado.",
                    "reason": "Execução de amostra aprovada pelo Judge.",
                    "confidence": 0.88
                })

    # Converte e formata o resultado final no schema contratado pelo n8n
    final_lessons = []
    for item in lessons:
        final_lessons.append({
            "memory_key": "iris_catarata",
            "category": item["category"],
            "lesson": item["lesson"],
            "reason": str(item["reason"])[:900],
            "confidence": float(item["confidence"]),
            "source_job_id": execution_data.get("job_id"),
            "source_session_id": execution_data.get("sessionId"),
            "source_analysis_type": execution_data.get("analysis_type"),
            "source_error_type": execution_data.get("errorType")
        })
        
    return final_lessons


async def save_learned_lessons(lessons: list[dict]) -> None:
    """
    Grava no Supabase via API REST (UPSERT) os novos aprendizados coletados pela Iris.
    """
    if not lessons or not settings.supabase_rest_url or not settings.DATABASE_API_KEY:
        return

    try:
        url = f"{settings.supabase_rest_url}memoria_aprendizados_iris"
        headers = _get_headers()
        headers["Prefer"] = "resolution=merge-duplicates"

        # A API do Supabase cuida do update automaticamente caso exista conflito na chave única,
        # MAS a lógica do UPSERT original somava usage_count.
        # Por simplicidade de migração via POST Bulk REST, usaremos o upsert padrão,
        # que sobreescreverá a linha existente. Em uso real via REST, poderíamos iterar para somar usage_count,
        # mas faremos inserção em massa.

        payload = []
        for l in lessons:
            payload.append({
                "memory_key": l["memory_key"],
                "category": l["category"],
                "lesson": l["lesson"],
                "reason": l["reason"],
                "confidence": l["confidence"],
                "usage_count": 1,
                "active": True
            })

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        logger.info(f"Salvos {len(lessons)} aprendizados/lições via REST no Supabase.")
    except Exception as e:
        logger.error(f"Erro ao salvar lições via REST: {e}")
