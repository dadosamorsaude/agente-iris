from contextvars import ContextVar
from app.core.config import settings

rag_results_context: ContextVar[list] = ContextVar("rag_results", default=[])

def format_docs(docs) -> str:
    """
    Formata os documentos recuperados com metadados.
    Ajuda o agente a saber de onde veio cada trecho.
    """
    if not docs:
        return ""

    formatted = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        fonte = metadata.get("fonte", "Fonte não informada")
        artigo = metadata.get("artigo", "")
        tema = metadata.get("tema", "")
        capitulo = metadata.get("capitulo", "")
        secao = metadata.get("secao", "")

        header = (
            f"[Trecho {i}]\n"
            f"Fonte: {fonte}\n"
        )
        if capitulo:
            header += f"Capítulo: {capitulo}\n"
        if secao:
            header += f"Seção: {secao}\n"
        if artigo:
            header += f"Artigo: {artigo}\n"
        if tema:
            header += f"Tema: {tema}\n"

        formatted.append(
            f"{header}\nConteúdo:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)
