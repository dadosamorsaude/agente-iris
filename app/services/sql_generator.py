from langchain_core.prompts import ChatPromptTemplate
from app.services.llm import get_chat_model_openai
from app.core.config import settings
import re


SQL_SYSTEM_PROMPT = """
Você é um especialista em gerar SQL seguro para AWS Athena.

Regras obrigatórias:
- Tabela: pdgt_amorsaude_inteligencia.tb_qualidade_prontuarios
- Nunca use SELECT *
- Use apenas colunas permitidas
- Sempre filtre por data_atendimento
- Nunca consulte datas futuras
- Prefira agregações (COUNT, SUM, AVG)
- Limite saídas detalhadas a 20 linhas
- Nunca use dados sensíveis (nome_paciente, CPF, RG, etc)
- Exclua obrigatoriamente id_especialidade IN (932, 1154, 993, 776, 777, 892, 1013, 711, 778, 658, 712, 732, 680, 1274, 779).

Colunas permitidas:
id_agendamento, id_atendimento, data_atendimento, status_agendamento,
id_procedimento, id_especialidade, especialidade, anamnese, conduta,
hipotese_diagnostica, observacao, orientacao, solicitacao,
especialidade_destino, cid_codigo, cid_descricao_detalhada,
id_clinica, clinica, regional, uf, id_profissional, nome_profissional,
prontuario_assinado

Retorne APENAS o código SQL puro, sem explicações.
"""


def _clean_sql(sql: str) -> str:
    """Removes markdown code blocks and extra whitespace."""
    # Remove ```sql ... ``` or ``` ... ```
    sql = re.sub(r"```(?:sql)?\n?(.*?)\n?```", r"\1", sql, flags=re.DOTALL)
    return sql.strip()


def generate_sql(message: str, hoje: str, ontem: str) -> str:
    llm = get_chat_model_openai(temperature=0.0, model=settings.MODEL_NAME_SQL)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SQL_SYSTEM_PROMPT),
            (
                "human",
                """
Pergunta do usuário: {message}

Use a regra de data:
WHERE date(data_atendimento) >= DATE '{ontem}'
AND date(data_atendimento) < DATE '{hoje}'
""",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "message": message,
            "hoje": hoje,
            "ontem": ontem,
        }
    )

    raw_sql = result.content or ""
    return _clean_sql(raw_sql)