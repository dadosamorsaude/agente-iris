import psycopg
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg.connect(DATABASE_URL, prepare_threshold=None)
cur = conn.cursor()

# 1. Total e todos os registros
cur.execute("SELECT id, user_id, created_at, question, score, approved FROM evaluation_logs ORDER BY created_at DESC;")
rows = cur.fetchall()
print(f"Total de registros na tabela: {len(rows)}")
print()
for r in rows:
    print(f"ID={r[0]} | user={r[1]} | data={r[2]} | score={r[3]} | aprovado={r[4]}")
    print(f"  Pergunta: {str(r[5])[:80]}")
    print()

# 2. Últimos 2 dias
cur.execute("SELECT COUNT(*) FROM evaluation_logs WHERE created_at >= NOW() - INTERVAL '2 days';")
print(f"Avaliacoes nos ultimos 2 dias: {cur.fetchone()[0]}")

# 3. Últimos 30 dias
cur.execute("SELECT COUNT(*) FROM evaluation_logs WHERE created_at >= NOW() - INTERVAL '30 days';")
print(f"Avaliacoes nos ultimos 30 dias: {cur.fetchone()[0]}")

# 4. Estrutura da tabela
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='evaluation_logs' ORDER BY ordinal_position;")
cols = cur.fetchall()
print()
print("Estrutura da tabela:")
for c in cols:
    print(f"  {c[0]}: {c[1]}")

conn.close()
