from typing import List, Dict
import asyncio
from app.services.llm import get_chat_model_openai
from app.core.logger import logger
import json

class PerformanceAuditSkill:
    """
    Skill para Auditoria de Performance Clínica em Lote.
    Analisa múltiplos prontuários e gera indicadores de qualidade agregados.

    Modelo: gpt-4.1 (via settings.MODEL_NAME) — temperatura 0, saída JSON
    estruturada com checklist CFM/RDC. Execução paralela via asyncio.gather().
    """

    def __init__(self):
        self.llm = get_chat_model_openai(temperature=0)

    async def audit_record(self, record_text: str) -> Dict:
        """Audita um único prontuário individualmente."""
        prompt = (
            "Aja como um auditor médico sênior. Analise o prontuário abaixo e responda APENAS um JSON com:\n"
            "1. 'compliant' (boolean): se atende às normas CFM/RDC.\n"
            "2. 'score' (int 0-100): nota de qualidade.\n"
            "3. 'missing_items' (list): o que faltou.\n"
            "4. 'risks' (list): riscos legais ou clínicos identificados.\n\n"
            f"Prontuário:\n{record_text}"
        )
        
        try:
            response = await self.llm.ainvoke(prompt)
            # Tenta limpar o texto se a IA retornar markdown
            clean_content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"Erro ao auditar prontuário individual: {e}")
            return {"compliant": False, "score": 0, "error": str(e)}

    async def run_batch_audit(self, records: List[str]) -> Dict:
        """
        Executa a auditoria em um lote de prontuários em paralelo.
        """
        logger.info(f"Iniciando auditoria em lote para {len(records)} prontuários.")
        
        # Executa em paralelo para ganhar tempo
        tasks = [self.audit_record(rec) for rec in records]
        results = await asyncio.gather(*tasks)
        
        # Agregação de Resultados
        total_records = len(results)
        compliant_count = sum(1 for r in results if r.get("compliant", False))
        average_score = sum(r.get("score", 0) for r in results) / total_records if total_records > 0 else 0
        
        # Consolidação de Falhas comuns
        all_missing_items = []
        for r in results:
            all_missing_items.extend(r.get("missing_items", []))
        
        # Conta frequência de itens faltantes
        frequency_map = {}
        for item in all_missing_items:
            frequency_map[item] = frequency_map.get(item, 0) + 1
            
        # Ordena as falhas mais comuns
        top_failures = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)[:5]

        report = {
            "summary": {
                "total_analyzed": total_records,
                "compliance_rate": f"{(compliant_count/total_records)*100:.1f}%",
                "average_quality_score": round(average_score, 1),
            },
            "top_compliance_issues": [
                {"item": item, "occurrences": count} for item, count in top_failures
            ],
            "recommendation": self._generate_manager_recommendation(average_score, top_failures)
        }
        
        return report

    def _generate_manager_recommendation(self, score: float, failures: List) -> str:
        if score > 90:
            return "Excelente performance. Manter treinamentos de rotina."
        elif score > 70:
            return "Boa performance, mas atenção aos itens faltantes recorrentes."
        else:
            return "Necessária intervenção imediata e treinamento da equipe nos protocolos de conformidade."

# Singleton para uso no projeto
performance_audit_skill = PerformanceAuditSkill()
