from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings
from app.core.observability import get_langsmith_callbacks


def get_chat_model_openai(temperature: float = None, model: str = None):
    return ChatOpenAI(
        model=model if model is not None else settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
        callbacks=get_langsmith_callbacks(),
    )

def get_chat_model_claude(temperature: float = None, model: str = None):
    return ChatAnthropic(
        model=model if model is not None else settings.MODEL_CLAUDE,
        temperature=temperature if temperature is not None else settings.TEMPERATURE_CLAUDE,
        api_key=settings.ANTHROPIC_API_KEY,
        callbacks=get_langsmith_callbacks(),
    )
