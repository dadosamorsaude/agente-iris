from contextvars import ContextVar
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langsmith import traceable
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


def get_compliance_retriever(namespace: str, k: int = 5):
    return get_retriever(
        index_name=settings.PINECONE_INDEX_CFM,
        namespace=namespace,
        k=k
    )


def get_pop_retriever(namespace: str = "", k: int = 5):
    return get_retriever(
        index_name=settings.PINECONE_INDEX_POP,
        namespace=namespace,
        k=k
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


@tool
@traceable(name="search_medical_compliance_tool")
def search_medical_compliance_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta para buscar diretrizes CFM, Resolução CFM 2.153/2016,
    regras de negócio do dashboard de qualidade, critérios de conformidade documental,
    anamnese, conduta, hipótese diagnóstica, CID, prontuário assinado, IQRC,
    boas práticas de registro clínico e normas RDC/ANVISA relacionadas à qualidade,
    segurança do paciente, serviços de saúde, odontologia, resíduos, infraestrutura
    e processamento/esterilização.
    """

    retriever_cfm = get_compliance_retriever(
        namespace="cfm_2153_2016",
        k=4
    )

    retriever_regras = get_compliance_retriever(
        namespace="regras_negocio_prontuario",
        k=4
    )

    if not retriever_cfm or not retriever_regras:
        return "Erro ao configurar buscador de conformidade."

    docs_cfm = retriever_cfm.invoke(query)
    docs_regras = retriever_regras.invoke(query)

    all_docs = docs_cfm + docs_regras

    captured = rag_results_context.get([])

    rag_results_context.set(
        captured + [
            {
                "source": "CFM",
                "namespace": "cfm_2153_2016",
                "query": query,
                "chunks": [d.page_content for d in docs_cfm],
                "metadata": [d.metadata for d in docs_cfm],
            },
            {
                "source": "Regras de Negócio",
                "namespace": "regras_negocio_prontuario",
                "query": query,
                "chunks": [d.page_content for d in docs_regras],
                "metadata": [d.metadata for d in docs_regras],
            },
        ]
    )

    if not all_docs:
        return "Nenhuma diretriz ou regra de negócio encontrada."

    return format_docs(all_docs)


@tool
@traceable(name="search_sop_tool")
def search_sop_tool(query: str) -> str:
    """
    OBRIGATÓRIO: Use esta ferramenta apenas para criação, revisão, estruturação
    ou elaboração de POPs, Procedimento Operacional Padrão, arquitetura de POPs,
    modelos de procedimento, instruções operacionais e documentos internos de processo.
    """

    retriever_pop = get_pop_retriever(namespace="", k=4)
    retriever_rdc = get_pop_retriever(namespace="rdc_anvisa", k=4)

    if not retriever_pop or not retriever_rdc:
        return "Erro ao configurar buscador de POPs."

    docs_pop = retriever_pop.invoke(query)
    docs_rdc = retriever_rdc.invoke(query)

    all_docs = docs_pop + docs_rdc

    captured = rag_results_context.get([])

    rag_results_context.set(
        captured + [
            {
                "source": "POP Interno",
                "query": query,
                "chunks": [d.page_content for d in docs_pop],
                "metadata": [d.metadata for d in docs_pop],
            },
            {
                "source": "RDC (Base para POP)",
                "query": query,
                "chunks": [d.page_content for d in docs_rdc],
                "metadata": [d.metadata for d in docs_rdc],
            }
        ]
    )

    if not all_docs:
        return "Nenhum POP ou RDC de base encontrado."

    return format_docs(all_docs)