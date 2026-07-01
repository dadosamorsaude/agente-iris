"""Stubs leves para permitir importar app.agent.specialists.sql_analyst em testes
sem instalar a stack pesada (langchain, langchain_openai, pinecone, etc.).
"""
import os
import sys
import types
import importlib

# Variaveis minimas para o Settings carregar.
os.environ.setdefault("OPENAI_API_KEY", "stub")
os.environ.setdefault("ANTHROPIC_API_KEY", "stub")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "stub")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "stub")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("ATHENA_DATABASE", "stub")
os.environ.setdefault("ATHENA_S3_STAGING_DIR", "s3://stub")
os.environ.setdefault("AGENTE_API_KEY", "stub")
os.environ.setdefault("ALLOWED_ORIGINS", "*")


def _mk(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# --- external libs we don't want to install in CI ---
def _tool(fn=None, *args, **kwargs):
    if callable(fn):
        async def ainvoke(arg=None, **_):
            return await fn(**(arg or {}))
        fn.ainvoke = ainvoke
        return fn
    def wrap(f):
        async def ainvoke(arg=None, **_):
            return await f(**(arg or {}))
        f.ainvoke = ainvoke
        return f
    return wrap


# --- external libs mock only if not installed ---
try:
    import langchain_core
except ImportError:
    sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    _mk("langchain_core.tools", tool=_tool, BaseTool=object)
    _mk("langchain_core.messages", HumanMessage=object, SystemMessage=object,
        AIMessageChunk=object, AIMessage=object, BaseMessage=object)

try:
    import pyathena
except ImportError:
    _mk("pyathena", connect=lambda **k: None)

try:
    import langchain_openai
except ImportError:
    _mk("langchain_openai", ChatOpenAI=object, OpenAIEmbeddings=object)

try:
    import langchain_anthropic
except ImportError:
    _mk("langchain_anthropic", ChatAnthropic=object)

try:
    import pinecone
except ImportError:
    _mk("pinecone", Pinecone=object)

try:
    import langchain_pinecone
except ImportError:
    _mk("langchain_pinecone", PineconeVectorStore=object)

try:
    import langgraph
except ImportError:
    sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    _mk("langgraph.prebuilt", create_react_agent=lambda *a, **k: None)

if "langsmith" not in sys.modules:
    def _traceable(*a, **k):
        def deco(fn):
            return fn
        if a and callable(a[0]):
            return a[0]
        return deco
    _mk("langsmith", traceable=_traceable)


# --- stub apenas SUBMODULOS internos pesados, NUNCA `app` em si ---
def _passthrough_traceable(*a, **k):
    def deco(fn):
        return fn
    if a and callable(a[0]):
        return a[0]
    return deco

_mk("app.core.observability",
    get_langsmith_callbacks=lambda: [],
    traceable=_passthrough_traceable,
    flush_langsmith=lambda: None,
    configure_langsmith=lambda: None,
)


class _DummyLLM:
    async def ainvoke(self, *a, **k):
        class R:
            content = "{}"
        return R()
_mk("app.services.llm",
    get_chat_model_openai=lambda *a, **k: _DummyLLM(),
    get_chat_model_claude=lambda *a, **k: _DummyLLM(),
)
