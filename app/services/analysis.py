from langchain_core.prompts import ChatPromptTemplate
from app.services.llm import get_chat_model_openai
from app.services.llm import get_chat_model_claude


ANALYSIS_SYSTEM_PROMPT = """
Você é Iris, assistente de análise clínica de cirurgias de catarata.

Responda sempre em português do Brasil.

Objetivo:
- analisar qualidade e conformidade dos registros clínicos de catarata (foco em: acuidade visual, biometria, técnica cirúrgica, complicações e desfecho)
- considerar campos com "xxx", "--", "ok", "NA" ou textos genéricos como **NÃO PREENCHIDOS**
- basear-se apenas nos dados fornecidos
- não alucinar
- usar linguagem cautelosa:
  "há indícios", "os registros sugerem", "pode haver oportunidade"

Se não houver dados suficientes, diga isso claramente.
"""


def analyze_data(message: str, data) -> str:
    llm = get_chat_model_claude(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYSIS_SYSTEM_PROMPT),
            (
                "human",
                """
Pergunta original:
{message}

Dados retornados:
{data}

Gere uma resposta clara, objetiva e estruturada.
""",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"message": message, "data": data})
    return result.content or ""