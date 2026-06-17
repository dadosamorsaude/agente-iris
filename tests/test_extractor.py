import pytest
from app.services.extractor import extract_cpf_regex, extract_clinical_analysis

def test_extract_cpf_regex():
    assert extract_cpf_regex("O CPF do paciente é 123.456.789-10") == "123.456.789-10"
    assert extract_cpf_regex("Paciente tem CPF 12345678910") == "12345678910"
    assert extract_cpf_regex("Nenhum cpf informado no prontuário.") is None
    assert extract_cpf_regex(None) is None
    assert extract_cpf_regex("") is None


@pytest.mark.asyncio
async def test_extract_clinical_analysis_mocked(monkeypatch):
    class MockModelDump:
        def model_dump(self):
            return {
                "classificacao": "positivo",
                "cpf": "123.456.789-10",
                "termo_gatilho": "facectomia OD",
                "evidencia_textual": "paciente indicado para facectomia OD",
                "campo_origem": "conduta",
                "lateralidade": "OD",
                "tipo_termo": "procedimento",
                "verbo_acao": "indicar",
                "contexto": "indicação de facectomia",
                "confianca": 90
            }

    class MockStructuredLLM:
        async def ainvoke(self, prompt):
            return MockModelDump()

    class MockLLM:
        def with_structured_output(self, schema):
            return MockStructuredLLM()

    monkeypatch.setattr("app.services.extractor.get_chat_model_openai", lambda temperature=None: MockLLM())

    res = await extract_clinical_analysis(
        user_message="Paciente com catarata senil, indicado facectomia OD. CPF: 123.456.789-10",
        agent_response="Identificado indicação de cirurgia de catarata (facectomia) no olho direito (OD)."
    )

    assert res["classificacao"] == "positivo"
    assert res["cpf"] == "123.456.789-10"
    assert res["termo_gatilho"] == "facectomia OD"
    assert res["lateralidade"] == "OD"
    assert res["confianca"] == 90
