from contextvars import ContextVar
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

rag_results_context: ContextVar[list] = ContextVar("rag_results", default=[])


def get_retriever(index_name: str, namespace: str = "", k: int = 5):
    """
    Inicializa e retorna um retriever Pinecone para um namespace específico.
    """

    if not settings.PINECONE_API_KEY:
        return None

    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-large",
        dimensions=3072
    )

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
        text_key="text"
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )


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
