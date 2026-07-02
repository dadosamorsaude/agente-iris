import logging
import time
from typing import Any
from contextvars import ContextVar
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.core.config import settings

logger = logging.getLogger(__name__)

# Contextos por-task para auditoria do LLM-as-Judge
athena_results_context: ContextVar[list] = ContextVar("athena_results", default=[])
rag_results_context: ContextVar[list] = ContextVar("rag_results", default=[])

# Cliente persistente em nível de módulo para reutilizar conexões SSE
_client = None

def get_mcp_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        servers = {
            "central": {
                "transport": "sse",
                "url": settings.CENTRAL_MCP_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.MCP_API_KEY}"
                } if hasattr(settings, "MCP_API_KEY") and settings.MCP_API_KEY else {}
            }
        }
        logger.info(f"Instanciando MultiServerMCPClient com transport 'sse' em {settings.CENTRAL_MCP_URL}")
        _client = MultiServerMCPClient(servers)
    return _client

async def invoke_mcp_tool(tool_name: str, input_args: dict[str, Any]) -> Any:
    """
    Conecta ao servidor MCP central, localiza a ferramenta solicitada e a executa.
    """
    client = get_mcp_client()
    tools = await client.get_tools()
    
    mcp_tool = None
    for t in tools:
        if t.name == tool_name or t.name.endswith(f"_{tool_name}"):
            mcp_tool = t
            break

    if not mcp_tool:
        raise ValueError(f"Ferramenta '{tool_name}' não encontrada no servidor MCP.")

    logger.info(f"Executando tool '{mcp_tool.name}' via MCP...")
    return await mcp_tool.ainvoke(input_args)


# ══════════════════════════════════════════════════════════════════════════════
# Cache de Prompts com TTL
# ══════════════════════════════════════════════════════════════════════════════

class PromptCache:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = (time.time(), value)

_prompt_cache = PromptCache(ttl_seconds=600)  # Cache de 10 minutos


async def get_cached_system_prompt(agent_id: str, data_hoje: str, data_ontem: str) -> str:
    """
    Busca o prompt de sistema a partir do cache local ou consulta o MCP caso expirado/ausente.
    """
    key = f"{agent_id}:{data_hoje}:{data_ontem}"
    cached = _prompt_cache.get(key)
    if cached is not None:
        logger.info("Retornando prompt do sistema a partir do cache local.")
        return cached

    client = get_mcp_client()
    logger.info(f"Buscando prompt de sistema para '{agent_id}' no servidor MCP...")
    messages = await client.get_prompt(
        server_name="central",
        prompt_name="setup_agent",
        arguments={
            "agent_id": agent_id,
            "data_hoje": data_hoje,
            "data_ontem": data_ontem
        }
    )
    if not messages:
        raise ValueError("O servidor MCP retornou uma lista de prompts vazia.")
    
    prompt_content = messages[0].content
    _prompt_cache.set(key, prompt_content)
    return prompt_content


async def get_cached_sql_expert_prompt(agent_id: str) -> str:
    """
    Busca o prompt do especialista de SQL a partir do cache local ou consulta o MCP caso expirado/ausente.
    """
    key = f"sql_expert:{agent_id}"
    cached = _prompt_cache.get(key)
    if cached is not None:
        logger.info("Retornando prompt do especialista SQL a partir do cache local.")
        return cached

    client = get_mcp_client()
    logger.info(f"Buscando prompt de especialista SQL para '{agent_id}' no servidor MCP...")
    messages = await client.get_prompt(
        server_name="central",
        prompt_name="build_sql_expert_prompt",
        arguments={
            "agent_id": agent_id
        }
    )
    if not messages:
        raise ValueError("O servidor MCP retornou uma lista de prompts vazia.")
    
    prompt_content = messages[0].content
    _prompt_cache.set(key, prompt_content)
    return prompt_content


# ══════════════════════════════════════════════════════════════════════════════
# Ferramentas Dinâmicas MCP com Auto-Binding de agent_id
# ══════════════════════════════════════════════════════════════════════════════

class MCPAthenaTool(BaseTool):
    name: str = "query_athena_tool"
    description: str = (
        "Executa consultas SQL no AWS Athena para análise de prontuários médicos. "
        "A query deve ser compatível com Presto/Athena. Retorne apenas dados relevantes."
    )

    class AthenaInput(BaseModel):
        sql: str = Field(description="Consulta SQL a ser executada")

    args_schema: type[BaseModel] = AthenaInput

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async execution")

    async def _arun(self, sql: str) -> str:
        import json
        
        # Validação de segurança básica de SQL
        sql_upper = sql.upper()
        forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE "]
        if any(token in sql_upper for token in forbidden):
            return "Consulta inválida: SQL contém operação proibida. Apenas SELECT é permitido."
        if "SELECT *" in sql_upper:
            return "Consulta inválida: SELECT * não é permitido. Por favor, liste as colunas explicitamente."

        response_obj = await invoke_mcp_tool("query_athena_tool", {"sql": sql, "agent_id": settings.AGENT_ID})
        
        raw_text = ""
        if isinstance(response_obj, list):
            parts = []
            for item in response_obj:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            raw_text = "".join(parts)
        elif isinstance(response_obj, str):
            raw_text = response_obj
        else:
            raw_text = str(response_obj)

        payload = {}
        try:
            payload = json.loads(raw_text)
        except Exception:
            payload = raw_text

        results = []
        if isinstance(payload, dict):
            results = payload.get("rows", [])
            hit_limit = payload.get("row_limit_hit", False)
            captured = athena_results_context.get([])
            athena_results_context.set(
                captured + [{"sql": sql, "results": results, "row_limit_hit": hit_limit}]
            )
            return json.dumps(results, default=str, ensure_ascii=False)
        elif isinstance(payload, list):
            results = payload
            captured = athena_results_context.get([])
            athena_results_context.set(
                captured + [{"sql": sql, "results": results, "row_limit_hit": False}]
            )
            return raw_text

        return raw_text


class MCPRAGTool(BaseTool):
    name: str
    description: str
    namespace_keys: list[str]

    class RAGInput(BaseModel):
        query: str = Field(description="Texto da busca semântica")

    args_schema: type[BaseModel] = RAGInput

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async execution")

    async def _arun(self, query: str) -> str:
        import asyncio
        results_text = []
        captured_chunks = []
        
        async def fetch_namespace(ns):
            response = await invoke_mcp_tool(
                "search_rag_tool",
                {
                    "query": query,
                    "agent_id": settings.AGENT_ID,
                    "namespace_key": ns,
                    "k": 4
                }
            )
            return ns, response

        tasks = [fetch_namespace(ns) for ns in self.namespace_keys]
        task_results = await asyncio.gather(*tasks)
        
        for namespace, response in task_results:
            raw_text = ""
            if isinstance(response, list):
                parts = []
                for item in response:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                    else:
                        parts.append(str(item))
                raw_text = "".join(parts)
            elif isinstance(response, str):
                raw_text = response
            else:
                raw_text = str(response)
                
            results_text.append(f"=== {namespace.upper()} ===\n{raw_text}")
            captured_chunks.append(raw_text)
            
        # Atualiza o contexto do RAG
        captured = rag_results_context.get([])
        new_captures = []
        for ns, chunk in zip(self.namespace_keys, captured_chunks):
            new_captures.append({
                "source": f"RAG ({ns})",
                "namespace": ns,
                "query": query,
                "chunks": [chunk],
                "metadata": []
            })
        rag_results_context.set(captured + new_captures)
        
        return "\n\n".join(results_text)


# Instanciação das tools dinâmicas expostas
query_athena_tool = MCPAthenaTool()
