# -*- coding: utf-8 -*-
"""
Script de Carga Historica -- Indexacao de Prontuarios no Pinecone

Uso:
    uv run python scripts/run_historical_index.py
    uv run python scripts/run_historical_index.py --mode d1
    uv run python scripts/run_historical_index.py --mode range --start 2026-01-01 --end 2026-06-17
"""

import asyncio
import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Root do projeto no path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configura o logger do indexador
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("indexer.log", encoding="utf-8")],
)
# Muta as bibliotecas de terceiros muito barulhentas
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("pyathena").setLevel(logging.WARNING)
logging.getLogger("app.services.prontuario_indexer").setLevel(logging.INFO)


from dotenv import load_dotenv
load_dotenv(override=True)

from app.services.prontuario_indexer import index_yesterday, index_date_range
from app.core.config import settings


def p(msg: str = "", flush: bool = True):
    """Print com flush imediato para evitar buffering no Windows."""
    print(msg, flush=flush)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


async def run_historico():
    """Carga historica dos ultimos 90 dias em janelas de 7 dias."""
    end = date.today()
    start = end - timedelta(days=90)

    p("=" * 60)
    p("  Iris -- Indexacao Historica 90 dias")
    p("=" * 60)
    p(f"  Periodo  : {start} -> {end}")
    p(f"  Estimado : ~405.000 atendimentos | ~$26 USD")
    p(f"  Logs     : indexer.log (neste diretorio)")
    p("=" * 60)
    p()

    # Gera todas as janelas de 7 dias
    windows = []
    w_start = start
    while w_start < end:
        w_end = min(w_start + timedelta(days=7), end)
        windows.append((w_start.isoformat(), w_end.isoformat()))
        w_start = w_end

    total_janelas = len(windows)
    grand_total = {"indexed": 0, "skipped": 0, "errors": 0, "total_fetched": 0}

    t_global = time.time()

    for i, (w_start, w_end) in enumerate(windows, start=1):
        p(f"  [{i:02d}/{total_janelas}] Janela: {w_start} -> {w_end} ...", flush=True)
        t_janela = time.time()

        result = await index_date_range(start_date=w_start, end_date=w_end, show_progress=True)

        dur = fmt_duration(time.time() - t_janela)
        p(f"         Buscados={result.get('total_fetched',0):,} | "
          f"Indexados={result.get('indexed',0):,} | "
          f"Ignorados={result.get('skipped',0):,} | "
          f"Erros={result.get('errors',0):,} | "
          f"Tempo={dur}")

        for key in grand_total:
            grand_total[key] += result.get(key, 0)

        # Pausa entre janelas
        if i < total_janelas:
            await asyncio.sleep(2)

    dur_total = fmt_duration(time.time() - t_global)

    p()
    p("=" * 60)
    p("  RESULTADO FINAL")
    p("=" * 60)
    p(f"  Janelas    : {total_janelas}")
    p(f"  Buscados   : {grand_total['total_fetched']:,}")
    p(f"  Indexados  : {grand_total['indexed']:,}")
    p(f"  Ignorados  : {grand_total['skipped']:,}")
    p(f"  Erros      : {grand_total['errors']:,}")
    p(f"  Duracao    : {dur_total}")
    p("=" * 60)

    if grand_total["errors"] == 0:
        p()
        p("  OK - Indexacao concluida com sucesso!")
        p("  Namespace 'prontuarios_pacientes' disponivel no Pinecone.")
    else:
        p()
        p(f"  AVISO: {grand_total['errors']:,} erros. Consulte indexer.log para detalhes.")
    p()

    return grand_total


async def run_d1():
    """Indexa apenas o dia anterior (D-1)."""
    yesterday = date.today() - timedelta(days=1)
    today = date.today()

    p("=" * 60)
    p("  Iris -- Indexacao D-1")
    p("=" * 60)
    p(f"  Periodo  : {yesterday} -> {today}")
    p(f"  Logs     : indexer.log")
    p("=" * 60)
    p()
    p("  Executando...", flush=True)

    t = time.time()
    result = await index_yesterday()
    dur = fmt_duration(time.time() - t)

    p()
    p("=" * 60)
    p("  RESULTADO")
    p("=" * 60)
    p(f"  Buscados   : {result.get('total_fetched', 0):,}")
    p(f"  Indexados  : {result.get('indexed', 0):,}")
    p(f"  Ignorados  : {result.get('skipped', 0):,}")
    p(f"  Erros      : {result.get('errors', 0):,}")
    p(f"  Duracao    : {dur}")
    p("=" * 60)
    p()

    return result


async def run_range(start_date: str, end_date: str):
    """Indexa um periodo customizado."""
    p("=" * 60)
    p("  Iris -- Indexacao Periodo Customizado")
    p("=" * 60)
    p(f"  Periodo  : {start_date} -> {end_date}")
    p(f"  Logs     : indexer.log")
    p("=" * 60)
    p()
    p("  Executando...", flush=True)

    t = time.time()
    result = await index_date_range(start_date=start_date, end_date=end_date, show_progress=True)
    dur = fmt_duration(time.time() - t)

    p()
    p("=" * 60)
    p("  RESULTADO")
    p("=" * 60)
    p(f"  Buscados   : {result.get('total_fetched', 0):,}")
    p(f"  Indexados  : {result.get('indexed', 0):,}")
    p(f"  Ignorados  : {result.get('skipped', 0):,}")
    p(f"  Erros      : {result.get('errors', 0):,}")
    p(f"  Duracao    : {dur}")
    p("=" * 60)
    p()

    return result


async def main():
    parser = argparse.ArgumentParser(description="Indexacao de prontuarios no Pinecone")
    parser.add_argument(
        "--mode",
        choices=["historico", "d1", "range"],
        default="historico",
        help="historico (90d), d1 (ontem), range (customizado)",
    )
    parser.add_argument("--start", help="Data inicial YYYY-MM-DD (modo range)")
    parser.add_argument("--end", help="Data final YYYY-MM-DD, exclusiva (modo range)")
    args = parser.parse_args()

    if args.mode == "range" and (not args.start or not args.end):
        p("Erro: --start e --end sao obrigatorios no modo range.")
        sys.exit(1)

    try:
        if args.mode == "historico":
            await run_historico()
        elif args.mode == "d1":
            await run_d1()
        else:
            await run_range(args.start, args.end)
    except KeyboardInterrupt:
        p()
        p("  Interrompido. Dados parcialmente indexados.")
        p("  Para retomar, use --mode range com a data onde parou.")
        sys.exit(0)
    except Exception as e:
        p(f"\n  ERRO FATAL: {e}")
        p("  Consulte indexer.log para o traceback completo.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
