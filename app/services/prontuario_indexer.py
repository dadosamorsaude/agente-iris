"""
Serviço de Indexação Contínua de Prontuários no Pinecone

Responsável por:
- Buscar prontuários do Athena (campos narrativos, anonimizados)
- Gerar embeddings via OpenAI text-embedding-3-large
- Fazer upsert no Pinecone com id_atendimento como chave idempotente
- Suportar carga histórica (90 dias) e incremental D-1

Anonimização: sem nome_paciente, sem cpf_paciente nos vetores.
Apenas id_paciente e id_atendimento como metadados identificadores.
"""

import asyncio
import logging
from datetime import date, timedelta

from app.core.clients import openai_async, pinecone_index
from app.core.config import settings
from app.tools.athena import _execute_athena_query

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
BATCH_SIZE = 100  # vetores por upsert Pinecone / textos por chamada OpenAI

# Campos narrativos para embedding (sem PII)
NARRATIVE_FIELDS = [
    ("Anamnese", "anamnese"),
    ("Conduta", "conduta"),
    ("Hipótese diagnóstica", "hipotese_diagnostica"),
    ("Solicitação", "solicitacao"),
    ("Prescrição", "prescricao"),
    ("Observação", "observacao"),
    ("CID", "cid_codigo"),
    ("Descrição CID", "cid_descricao_detalhada"),
]

# Metadados armazenados no Pinecone (sem PII)
METADATA_FIELDS = [
    "id_atendimento",
    "id_paciente",
    "data_atendimento",
    "clinica",
    "regional",
    "uf",
    "municipio",
    "especialidade",
]

# Colunas buscadas no Athena
_ATHENA_COLUMNS = [
    "id_atendimento",
    "id_paciente",
    "data_atendimento",
    "especialidade",
    "anamnese",
    "conduta",
    "hipotese_diagnostica",
    "observacao",
    "solicitacao",
    "prescricao",
    "cid_codigo",
    "cid_descricao_detalhada",
    "clinica",
    "regional",
    "uf",
    "municipio",
]


def _get_pinecone_index():
    """Retorna o índice Pinecone configurado (singleton)."""
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY não configurada.")
    return pinecone_index(settings.PINECONE_RAG_INDEX)


def _build_text(row: dict) -> str:
    """
    Constrói o texto anonimizado para embedding a partir dos campos narrativos.
    Exclui nome_paciente e cpf_paciente por design (conformidade LGPD).
    """
    parts = []
    for label, field in NARRATIVE_FIELDS:
        value = row.get(field)
        if value and str(value).strip():
            parts.append(f"{label}: {str(value).strip()}")
    return "\n".join(parts)


def _build_metadata(row: dict) -> dict:
    """
    Extrai apenas os metadados permitidos (sem PII).
    Converte valores para string para compatibilidade com Pinecone.
    """
    metadata = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if value is not None:
            metadata[field] = str(value)
    return metadata


def _fetch_prontuarios(start_date: str, end_date: str) -> list[dict]:
    """
    Busca prontuários do Athena no período [start_date, end_date).
    Retorna apenas campos narrativos + metadados — sem nome_paciente, sem cpf_paciente.
    """
    cols = ",\n            ".join(_ATHENA_COLUMNS)
    sql = f"""
        SELECT
            {cols}
        FROM pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
        WHERE data_atendimento >= DATE '{start_date}'
          AND data_atendimento < DATE '{end_date}'
    """
    logger.info(f"Buscando prontuários do Athena: {start_date} → {end_date}")
    return _execute_athena_query(sql)


def _fetch_prontuarios_by_ids(ids_atendimento: list[str]) -> list[dict]:
    """Busca prontuários específicos por id_atendimento."""
    if not ids_atendimento:
        return []
    ids_str = ", ".join(ids_atendimento)
    cols = ",\n            ".join(_ATHENA_COLUMNS)
    sql = f"""
        SELECT
            {cols}
        FROM pdgt_amorsaude_tecnologia.fl_prontuarios_oftalmologia
        WHERE id_atendimento IN ({ids_str})
    """
    return _execute_athena_query(sql)


async def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Gera embeddings em batch via OpenAI text-embedding-3-large (singleton)."""
    response = await openai_async().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


async def index_batch(rows: list[dict], show_progress: bool = False) -> dict:
    """
    Indexa um batch de prontuários no Pinecone de forma concorrente.

    - Divide as linhas em sub-batches de tamanho BATCH_SIZE
    - Gera embeddings via OpenAI de forma paralela (limitado por Semaphore)
    - Faz upsert no Pinecone usando thread pool para não bloquear o loop de eventos
    - Imprime o progresso em tempo real se show_progress for True
    """
    if not rows:
        return {"indexed": 0, "skipped": 0, "errors": 0}

    index = _get_pinecone_index()
    
    # Divide em sub-batches de BATCH_SIZE
    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    total_rows = len(rows)
    
    # Semáforo para limitar concorrência (Render free tem 512MB de RAM,
    # cada batch carrega ~12MB de embeddings — segura em 2 paralelos).
    sem = asyncio.Semaphore(2)
    lock = asyncio.Lock()
    
    # Contadores globais do lote
    stats = {"indexed": 0, "skipped": 0, "errors": 0}
    processed_rows = 0

    async def process_sub_batch(batch_idx: int, batch: list[dict]):
        nonlocal processed_rows
        # Filtra registros sem conteúdo narrativo
        texts, valid_rows = [], []
        batch_skipped = 0
        for row in batch:
            text = _build_text(row)
            if not text.strip():
                batch_skipped += 1
                continue
            texts.append(text)
            valid_rows.append(row)

        if not texts:
            async with lock:
                stats["skipped"] += batch_skipped
                processed_rows += len(batch)
                if show_progress and (processed_rows % 1000 == 0 or processed_rows == total_rows):
                    print(f"         Progresso: {processed_rows}/{total_rows} prontuários processados...", flush=True)
            return

        try:
            # Limita a concorrência na geração de embeddings
            async with sem:
                embeddings = await _generate_embeddings(texts)

            vectors = []
            for row, embedding in zip(valid_rows, embeddings):
                id_atendimento = str(row.get("id_atendimento", "")).strip()
                if not id_atendimento:
                    batch_skipped += 1
                    continue

                metadata = _build_metadata(row)
                metadata["text"] = _build_text(row)[:1000]

                vectors.append({
                    "id": id_atendimento,
                    "values": embedding,
                    "metadata": metadata,
                })

            if vectors:
                # Upsert é síncrono, então executamos em thread pool (asyncio.to_thread)
                async with sem:
                    await asyncio.to_thread(
                        index.upsert,
                        vectors=vectors,
                        namespace=settings.PINECONE_NS_PRONTUARIOS,
                    )

            async with lock:
                stats["indexed"] += len(vectors)
                stats["skipped"] += batch_skipped
                processed_rows += len(batch)
                if show_progress and (processed_rows % 1000 == 0 or processed_rows == total_rows):
                    print(f"         Progresso: {processed_rows}/{total_rows} prontuários processados...", flush=True)

        except Exception as e:
            logger.error(f"Erro ao indexar sub-batch [{batch_idx * BATCH_SIZE}:{(batch_idx + 1) * BATCH_SIZE}]: {e}")
            async with lock:
                stats["errors"] += len(texts)
                stats["skipped"] += batch_skipped
                processed_rows += len(batch)
                if show_progress and (processed_rows % 1000 == 0 or processed_rows == total_rows):
                    print(f"         Progresso: {processed_rows}/{total_rows} prontuários processados...", flush=True)

    # Executa todos os sub-batches de forma concorrente
    tasks = [process_sub_batch(idx, b) for idx, b in enumerate(batches)]
    await asyncio.gather(*tasks)

    return stats


async def index_date_range(start_date: str, end_date: str, show_progress: bool = False) -> dict:
    """
    Busca e indexa todos os prontuários de um período [start_date, end_date).

    Args:
        start_date: Data inicial ISO (YYYY-MM-DD), inclusiva
        end_date:   Data final ISO (YYYY-MM-DD), exclusiva
        show_progress: Se True, imprime progresso de processamento no terminal

    Returns:
        dict com chaves: indexed, skipped, errors, total_fetched
    """
    try:
        rows = await asyncio.to_thread(_fetch_prontuarios, start_date, end_date)
    except Exception as e:
        logger.error(f"Erro ao buscar prontuários do Athena ({start_date} → {end_date}): {e}")
        return {"indexed": 0, "skipped": 0, "errors": 1, "total_fetched": 0}

    total_fetched = len(rows)
    logger.info(f"Prontuários buscados: {total_fetched} | {start_date} → {end_date}")

    result = await index_batch(rows, show_progress=show_progress)
    result["total_fetched"] = total_fetched

    logger.info(
        f"Indexação concluída | fetched={total_fetched} "
        f"| indexed={result['indexed']} | skipped={result['skipped']} | errors={result['errors']}"
    )
    return result


async def index_yesterday() -> dict:
    """
    Indexa os prontuários do dia anterior (D-1).
    Função principal para uso no cron diário via endpoint /internal/index-prontuarios.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    logger.info(f"Indexação D-1: {yesterday}")
    return await index_date_range(start_date=yesterday, end_date=today)


async def index_historical_90_days() -> dict:
    """
    Carga histórica dos últimos 90 dias.
    Processa em janelas de 7 dias para evitar timeout do Athena e
    respeitar os rate limits da OpenAI Embeddings API.

    Custo estimado: ~$26 USD (405K atendimentos × 500 tokens × $0,13/M tokens).
    """
    end = date.today()
    start = end - timedelta(days=90)

    total: dict = {"indexed": 0, "skipped": 0, "errors": 0, "total_fetched": 0}

    window_start = start
    while window_start < end:
        window_end = min(window_start + timedelta(days=7), end)

        logger.info(f"Janela histórica: {window_start.isoformat()} → {window_end.isoformat()}")
        result = await index_date_range(
            start_date=window_start.isoformat(),
            end_date=window_end.isoformat(),
            show_progress=True,
        )
        for key in total:
            total[key] += result.get(key, 0)

        window_start = window_end

        # Pausa entre janelas para respeitar rate limits
        await asyncio.sleep(2)

    logger.info(f"Carga histórica 90 dias concluída: {total}")
    return total


async def index_batch_by_ids(ids_atendimento: list) -> dict:
    """
    Indexa prontuários específicos por id_atendimento.

    Usado no trigger on-demand do _post_execution da Iris:
    aproveita os IDs já retornados pelo Athena naquela execução
    para indexar em background, sem custo adicional de SQL.

    Idempotente: o upsert do Pinecone sobrescreve silenciosamente
    se o id_atendimento já existir no índice.
    """
    if not ids_atendimento:
        return {"indexed": 0, "skipped": 0, "errors": 0}

    str_ids = [str(i) for i in ids_atendimento if i]
    if not str_ids:
        return {"indexed": 0, "skipped": 0, "errors": 0}

    try:
        rows = await asyncio.to_thread(_fetch_prontuarios_by_ids, str_ids)
    except Exception as e:
        logger.error(f"Erro ao buscar prontuários por IDs: {e}")
        return {"indexed": 0, "skipped": 0, "errors": 1}

    return await index_batch(rows)
