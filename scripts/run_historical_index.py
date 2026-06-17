"""
Script de Carga Histórica — Indexação de Prontuários no Pinecone

Executa a indexação dos últimos 90 dias diretamente da sua máquina,
usando as credenciais do .env local.

Uso:
    uv run python scripts/run_historical_index.py

    # Ou com período customizado:
    uv run python scripts/run_historical_index.py --start 2026-01-01 --end 2026-06-17

    # Para indexar apenas D-1 (teste rápido):
    uv run python scripts/run_historical_index.py --mode d1
"""

import asyncio
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Garante que o root do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from app.services.prontuario_indexer import (
    index_historical_90_days,
    index_yesterday,
    index_date_range,
)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def print_header(mode: str, start: str = None, end: str = None):
    print()
    print("=" * 60)
    print("  Iris -- Indexacao de Prontuarios no Pinecone")
    print("=" * 60)
    if mode == "historico":
        inicio = date.today() - timedelta(days=90)
        fim = date.today()
        print(f"  Modo     : Carga Historica 90 dias")
        print(f"  Periodo  : {inicio} -> {fim}")
        print(f"  Estimado : ~405.000 atendimentos | ~$26 USD")
    elif mode == "d1":
        yesterday = date.today() - timedelta(days=1)
        print(f"  Modo     : D-1 (dia anterior)")
        print(f"  Periodo  : {yesterday}")
        print(f"  Estimado : ~4.500 atendimentos | ~$0.29 USD")
    elif mode == "range":
        print(f"  Modo     : Periodo Customizado")
        print(f"  Periodo  : {start} -> {end}")
    print("=" * 60)
    print()


def print_result(result: dict, duration: float):
    print()
    print("=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  Buscados   : {result.get('total_fetched', 0):,}")
    print(f"  Indexados  : {result.get('indexed', 0):,}")
    print(f"  Ignorados  : {result.get('skipped', 0):,}  (sem conteudo narrativo)")
    print(f"  Erros      : {result.get('errors', 0):,}")
    print(f"  Duracao    : {fmt_duration(duration)}")
    print("=" * 60)

    errors = result.get("errors", 0)
    if errors == 0:
        print("\n  OK - Indexacao concluida com sucesso!")
        print("     Namespace 'prontuarios_pacientes' disponivel no Pinecone.")
    else:
        print(f"\n  AVISO: Concluido com {errors} erros. Verifique os logs acima.")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Indexação de prontuários no Pinecone")
    parser.add_argument(
        "--mode",
        choices=["historico", "d1", "range"],
        default="historico",
        help="Modo de indexação: historico (90d), d1 (ontem), range (customizado)",
    )
    parser.add_argument("--start", help="Data inicial YYYY-MM-DD (modo range)")
    parser.add_argument("--end", help="Data final YYYY-MM-DD, exclusiva (modo range)")
    args = parser.parse_args()

    if args.mode == "range" and (not args.start or not args.end):
        print("Erro: --start e --end são obrigatórios no modo range.")
        sys.exit(1)

    print_header(args.mode, args.start, args.end)

    if args.mode == "historico":
        print("  Processando em janelas de 7 dias para respeitar os")
        print("  rate limits da OpenAI. Acompanhe o progresso nos logs:\n")

    start_time = time.time()

    try:
        if args.mode == "historico":
            result = await index_historical_90_days()
        elif args.mode == "d1":
            result = await index_yesterday()
        else:
            result = await index_date_range(
                start_date=args.start,
                end_date=args.end,
            )
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrompido pelo usuário. Dados parcialmente indexados.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Erro fatal: {e}")
        sys.exit(1)

    duration = time.time() - start_time
    print_result(result, duration)


if __name__ == "__main__":
    asyncio.run(main())
