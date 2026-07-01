import asyncio
import logging
from typing import Any, AsyncGenerator
from app.services.validator import validate_response

logger = logging.getLogger(__name__)

def extract_text_from_content(content: Any) -> str:
    """Extrai texto bruto de conteúdos e blocos complexos do LangChain/Anthropic."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "get"):
                if block.get("type") == "text":
                    text_parts.append(block.get("type", ""))
        return "".join(text_parts)
    return str(content)


async def stream_agent_response(
    agent: Any,
    input_messages: list,
    config: dict,
    message: str,
    stream: bool,
    tool_aliases: dict[str, str]
) -> AsyncGenerator[str, None]:
    """
    Consome o stream de eventos do LangGraph assincronamente.
    Envia keep-alives periódicos e gerencia a tradução de ferramentas para o usuário.
    """
    try:
        if stream:
            full_response = ""
            active_tools = 0
            event_queue = asyncio.Queue()

            async def consume_stream():
                try:
                    async for event in agent.astream_events(
                        {"messages": input_messages},
                        config=config,
                        version="v2",
                    ):
                        await event_queue.put(event)
                except Exception as e:
                    await event_queue.put(e)
                finally:
                    await event_queue.put(None)

            stream_task = asyncio.create_task(consume_stream())

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Envia espaço em branco Keep-Alive para manter conexão ativa no Render
                        yield " "
                        continue

                    if event is None:
                        break
                    if isinstance(event, Exception):
                        raise event

                    kind = event.get("event")

                    if kind == "on_tool_start":
                        active_tools += 1
                        tool_name = event.get("name", "ferramenta")
                        alias = tool_aliases.get(tool_name, tool_name)
                        if active_tools == 1:
                            logger.info(f"Executando ferramenta: {tool_name}")
                            yield f"\n[⚙️ Pensando: Acionando {alias}...]\n"
                        continue
                        
                    if kind == "on_tool_end":
                        active_tools -= 1
                        tool_name = event.get("name", "ferramenta")
                        alias = tool_aliases.get(tool_name, tool_name)
                        if active_tools == 0:
                            yield f"\n[✅ {alias} finalizado]\n"
                        continue

                    if kind == "on_chat_model_stream":
                        if active_tools == 0:
                            chunk = event.get("data", {}).get("chunk")
                            if not chunk:
                                continue
                            text = extract_text_from_content(getattr(chunk, "content", None))
                            if text:
                                full_response += text
                                yield text

            except asyncio.CancelledError:
                logger.warning("Streaming cancelado pelo cliente.")
                return

        else:
            result = await agent.ainvoke({"messages": input_messages}, config=config)

            messages = result.get("messages", [])
            if not messages:
                final_response = "Não foi possível gerar uma resposta."
            else:
                response_text = extract_text_from_content(messages[-1].content)
                final_response = response_text

            # Retry automático em caso de erro técnico
            if final_response.startswith("Erro técnico:"):
                logger.warning("Resposta com erro técnico — refazendo consulta...")
                result = await agent.ainvoke({"messages": input_messages}, config=config)
                messages = result.get("messages", [])
                if messages:
                    response_text = extract_text_from_content(messages[-1].content)
                    new_response = response_text
                    if not new_response.startswith("Erro técnico:"):
                        final_response = new_response
                        logger.info("Retry bem-sucedido")

            yield final_response

    except Exception as e:
        logger.exception("Erro no AgentExecutor")
        yield f"Erro técnico: {str(e)}"
