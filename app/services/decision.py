from langchain_core.prompts import ChatPromptTemplate
from app.services.llm import get_chat_model_openai
from app.services.llm import get_chat_model_claude
from app.services.schemas import DecisionOutput


SYSTEM_PROMPT = """
Você é um classificador de intenção.

Classifique a mensagem do usuário em exatamente uma das ações:
- analisar_prontuarios
- consultar_pop
- responder_direto
"""


def decide_action(message: str) -> dict:
    llm = get_chat_model_claude(temperature=0.0)
    structured_llm = llm.with_structured_output(DecisionOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Mensagem do usuário: {message}"),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"message": message})

    return result.model_dump()