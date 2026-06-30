import logging
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.core.config import settings

logger = logging.getLogger(__name__)

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
                }
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
