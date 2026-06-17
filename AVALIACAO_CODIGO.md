# Avaliação de Código — Projeto Iris

**Data:** 17/06/2026
**Escopo:** repositório completo, com foco em complexidade, retries, gestão de memória (Render Free 512 MB) e código morto.

---

## TL;DR — Top 10 ações por impacto

1. Remover `indexer.log` (141 MB) do working tree e adicionar `*.log` ao `.gitignore`.
2. Singleton de clientes HTTP/SDK (OpenAI, Anthropic, Pinecone, httpx) — hoje instanciados por request.
3. Adicionar `timeout` e `max_retries` em `services/transcription.py` e `services/llm.py`.
4. Limitar fallbacks in-memory (`services/memory.py`, `services/evaluation_store.py`) com `deque(maxlen=...)`.
5. Streaming de upload de áudio + TTL em `temp_audios/`.
6. Backoff exponencial + jitter no retry de SQL em `sql_analyst.py`.
7. Quebrar `sql_analyst.py` (1244 linhas) em 3+ módulos.
8. Deletar arquivos mortos: `chunking-*.py`, `criar_index_pinecone.py`, `test_catarata.py`, `tools/transcription.py`, `scratch/`, `test_caracterizacao_v2.py`.
9. Corrigir bug runtime em `evaluator.py` (usa `settings.MODEL_NAME` sem importar `settings`).
10. WebSocket em `ws.py` com `asyncio.wait_for` + heartbeat (workers presos no Render free).

---

## 1. Excesso de complexidade

### 1.1 `app/agent/specialists/sql_analyst.py` (1244 linhas) — risco ALTO

Arquivo mais crítico do repositório. Concentra três responsabilidades distintas e tem várias funções gigantes.

- `_extract_period` (linhas 198–406, ~208 linhas): mais de 20 ramificações `if/return`. Cada bloco temporal repete EXATAMENTE o mesmo dicionário de retorno (`status/start/end_exclusive/sql_filter`) 7 vezes. Extrair helper `_build_period_payload(start, end)` + tabela `(regex, handler)` reduz para ~80 linhas.
- `sql_analyst_expert` (1042–1244, ~200 linhas): aninhamento de 4 níveis (try/for attempt/try/except), com 3 blocos de retorno de erro quase idênticos (1113–1127, 1132–1143, 1150–1161). Extrair `_build_error_payload(error_type, message, sql, intent, …)`.
- `_format_sql_result` (920–1022): 3 caminhos paralelos retornando dicts com mesmas chaves. Unificar via `_make_payload(kind, …)`.
- `_resolve_query_shape` (409–498): 6 `return` quase idênticos. Substituir por dataclass `QueryShape` + tabela `mode -> shape`.
- `extract_laterality` (811–848): 7 regex compilados **a cada chamada**. Pré-compilar no módulo (`_OD_RE`, `_OE_RE`, `_AO_RE`) — função roda milhares de vezes em laços.
- `_build_query_shape_contract` (501–563): f-string de ~60 linhas. Mover para `app/agent/prompts/sql_contracts.py`.

**Sugestão estrutural:** dividir em `period_extractor.py`, `sql_prompt_builder.py`, `sql_executor.py`.

### 1.2 `app/api/iris_chat.py` (404 linhas)

- Branches stream/non-stream com duplicação massiva (47–143): mesma cadeia (cache → `run_iris_agent` → `extract_clinical_analysis` → captura de contextos → `background_tasks.add_task`) aparece duas vezes. Extrair `_handle_iris_response(req, background_tasks, streaming: bool)`.
- `report_endpoint` (151–352): ~200 linhas, ~12 branches. Decompor em `_resolve_period_range`, `_dedup_rows`, `_apply_report_filters`.
- Imports tardios dentro do handler: `from app.services.extractor import extract_clinical_analysis` aparece 3 vezes (55, 82, 118, 141). Mover para topo.
- `_clinics_cache` global manual (356–403): TTL 300s sem limite de tamanho. `pt_br_sort_key` (389–391) duplica `normalize_text` de `services/intent.py`.

### 1.3 `app/services/learning.py` (299 linhas)

- `generate_lessons_from_execution` (94–260, 166 linhas): 12 ramos `if judge_failed and X:` quase idênticos, cada um construindo um dict com 4 chaves. Forte candidato a tabela declarativa:
  ```python
  LESSON_RULES = [
      {"id": "amostra", "when": lambda d: d.judge_failed and d.is_sample,
       "category": "amostra", "lesson": "...", "confidence": 0.95},
      ...
  ]
  ```
  Reduz para <50 linhas.

### 1.4 `app/agent/iris_orchestrator.py`

- `extract_text_from_content` (73–92): 5 caminhos `isinstance` + `getattr` para tipos de `AIMessageChunk`. Frágil.
- Stream/non-stream com mesma lógica de detecção de tools repetida (172–197). Extrair `_extract_tools_used(messages_or_events)`.

### 1.5 `app/services/prontuario_indexer.py`

- `_fetch_prontuarios` e `_fetch_prontuarios_by_ids` (109–138) duplicam SQL via `_ATHENA_COLUMNS`. Unificar em `_build_select_sql(where_clause)`.
- `index_batch.process_sub_batch` (178–243) repete logging de progresso 3x.

---

## 2. Instruções e prompts repetidos

### 2.1 System prompts dispersos

Quatro prompts gigantes com regras sobrepostas (“nunca invente dados”, “não exponha SQL/arquitetura”, “use exatamente os números”):

- `iris_orchestrator.IRIS_SYSTEM_PROMPT` (30–71)
- `sql_analyst._generate_sql.system_prompt` (640–684)
- `clinical_rag.system_prompt` (58–111)
- `evaluator.EVALUATOR_SYSTEM_PROMPT` (22–63)

**Sugestão:** criar `app/agent/prompts/` com arquivos `.md`/`.txt` versionáveis e um `shared_safety_rules.md` referenciado pelos quatro. Tira 30+ linhas de string do código fonte.

### 2.2 Prompt de análise de voz duplicado

Bloco “Estruture o texto nos campos: ANAMNESE, CONDUTA…” idêntico em `api/voice.py:51–57` e `api/ws.py:90–96`. Extrair constante `VOICE_ANALYSIS_PROMPT` em `services/transcription.py`.

### 2.3 Regras de SQL repetidas dentro do próprio `sql_analyst.py`

`_generate_sql` (640–684) e `_repair_sql_for_execution` (775–802) repetem regras (`Nunca SELECT *`, `Use lower(coalesce(...))`) que **também já estão** dentro do `CATARATA_SCHEMA` (33–101). Três versões das mesmas regras em um arquivo. Extrair `SQL_RULES_BLOCK` constante.

### 2.4 `UPLOAD_DIR = "temp_audios"` hard-coded 4 vezes

`tools/transcription.py:10`, `api/ws.py:12`, `api/voice.py:12`, `api/audio.py:8`. Mover para `app/core/config.py` (`settings.UPLOAD_DIR`).

### 2.5 `_get_headers()` duplicado

Idêntico em `services/learning.py:12–19` e `services/evaluation_store.py:25–32` (auth Supabase). Criar `app/services/supabase_client.py` com `get_headers()` + helper `async post_jsonb(table, payload)`.

### 2.6 `compute_summary_stats` em `iris_chat.py` (263–284)

Replica em Python o cálculo que `sql_analyst._format_sql_result` (988–997) já produz no `summary`. Mesmas chaves. Mover para `app/services/clinical_summary.py`.

---

## 3. Retries e timeouts

### 3.1 CRÍTICO — `services/transcription.py:16-25`

Upload Whisper instancia `OpenAI(...)` por request (sem reuso de conexão) e roda síncrono no event loop sem `timeout`. Default do SDK é ~600s. Como é chamado direto de rotas async (`voice.py:43`, `ws.py:78`), bloqueia o único worker do Render free. Trocar por `AsyncOpenAI` global com `timeout=30`, `max_retries=2`.

### 3.2 CRÍTICO — `services/llm.py:7-21`

`ChatOpenAI` e `ChatAnthropic` instanciados a cada chamada, sem `timeout` nem `max_retries`. Sem reuso de `httpx.AsyncClient`. Singleton módulo-level com `timeout=30, max_retries=2`.

### 3.3 CRÍTICO — `sql_analyst.py:1163-1233`

`max_attempts = 3` para Athena com `_repair_sql_for_execution` (chamada LLM extra a cada falha) **sem `asyncio.sleep`, sem backoff, sem jitter**. Em incidente de Athena (throttling) as 3 tentativas em rajada amplificam o problema, gastam tokens com repair e prendem o worker por ~30–60s. Adicionar:

```python
await asyncio.sleep(2 ** attempt + random.random())
```

E distinguir erros transitórios vs lógicos (não vale repair de erro 4xx).

### 3.4 Alto — `services/memory.py:47-62`

Pool asyncpg sem `timeout` em `create_pool` (default infinito para aquisição). Em primeiro request após sleep do Render, janela longa. Adicionar `timeout=10`.

### 3.5 Alto — Chamadas Pinecone/OpenAI sem timeout explícito

`services/cache.py:30-44`, `tools/prontuario_search.py:65-72`, `services/prontuario_indexer.py:143-149`. Configurar `OpenAIEmbeddings(timeout=20, max_retries=2)`.

### 3.6 Alto — `httpx.AsyncClient` por chamada

`evaluation_store.py:68,146,163,213` e `learning.py:39,293` criam cliente novo a cada call ao Supabase (5 lugares). Cliente módulo-level no startup com `limits=httpx.Limits(max_connections=10)`.

### 3.7 Médio — `tools/athena.py:33-59`

Conexão pyathena por request, sem `query_execution_timeout`. Usar boto3 com `Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 2})`.

### 3.8 Médio — `api/ws.py:44-114`

Loop `while True` em WebSocket sem `asyncio.wait_for` no `receive`. Conexões zumbis seguram worker indefinidamente e mantém `temp_path` aberto. Envolver em `asyncio.wait_for(..., timeout=60)` e implementar ping-pong.

---

## 4. Gestão de memória (Render Free 512 MB)

### 4.1 CRÍTICO — `indexer.log` 141 MB no working tree

`scripts/run_historical_index.py:26` usa `FileHandler` sem rotação. `.gitignore` não inclui `*.log`. O arquivo já existe no disco do dev (não está commitado, mas atrapalha clones/builds locais). Em runtime no Render, o disco efêmero do free (~1 GB) pode encher.

**Ação imediata:**
- `rm indexer.log` (e `git rm --cached indexer.log` se em algum momento foi rastreado).
- Adicionar `*.log` ao `.gitignore` (já está no `.dockerignore`).
- Trocar `FileHandler` por `RotatingFileHandler(maxBytes=10_000_000, backupCount=2)` ou só logar para stdout (Render coleta automaticamente).

### 4.2 CRÍTICO — Fallback in-memory ilimitado

- `services/memory.py:13` — `_memory_store: dict = {}` guarda mensagens por session_id sem teto. Sem `DATABASE_URL`, cresce indefinidamente.
- `services/evaluation_store.py:22` — `_memory_store: list[dict] = []` ilimitado, anexa `raw_data` (linhas completas do Athena).

Trocar por `collections.deque(maxlen=20)` por session e `deque(maxlen=200)` global.

### 4.3 CRÍTICO — Clientes pesados instanciados por request

Múltiplos pontos criam `OpenAI()`, `AsyncOpenAI()`, `Pinecone()`, `ChatOpenAI()` a cada chamada. Cada instância carrega seu próprio `httpx.AsyncClient`, tiktoken, gRPC pool. Em 512 MB isso pesa.

**Sugestão:** criar `app/core/clients.py` com singletons:

```python
# app/core/clients.py
from functools import lru_cache
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pinecone import Pinecone
import httpx

@lru_cache(maxsize=1)
def openai_async() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=30, max_retries=2)

@lru_cache(maxsize=1)
def pinecone() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)

@lru_cache(maxsize=1)
def supabase_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.SUPABASE_URL,
        headers={...},
        timeout=10,
        limits=httpx.Limits(max_connections=10),
    )
```

### 4.4 Alto — Upload de áudio carrega tudo na RAM

`api/audio.py:32` faz `content = await file.read()` (arquivo inteiro). `api/voice.py:30-39` idem. Sem limite de tamanho — upload de 50 MB com 2 conexões concorrentes estoura 512 MB.

Streaming: `await file.read(64*1024)` em loop, direto para `tempfile.SpooledTemporaryFile`. Validar `Content-Length`.

### 4.5 Alto — `temp_audios/` sem limpeza

`voice.py` e `ws.py` apagam no `finally`, mas `audio.py` deixa arquivos órfãos até reboot. Adicionar job de TTL (apagar > 30 min) ou trocar por `tempfile.NamedTemporaryFile(delete=True)`.

### 4.6 Alto — Indexação histórica não pode rodar no Render free

`services/prontuario_indexer.py` divide em sub-batches de 100 com concorrência 10 (`Semaphore(10)`). Cada batch segura ~12 MB de embeddings + payloads JSON simultâneos. O endpoint `/internal/index-prontuarios/historico` foi exposto em `indexer_router.py`. **Mover para job externo** (GitHub Action já existe — manter só D-1 no Render, com `Semaphore(2)` e batch menor).

### 4.7 Médio — `_pool` nunca fechado no shutdown

`lifespan` em `main.py:19-35` não chama `close_pool()`. Em redeploys, conexões Supabase ficam abertas até timeout.

### 4.8 Médio — `PineconeSemanticCache()` instanciado no import

`services/cache.py:69-70` instancia no nível do módulo, criando `OpenAIEmbeddings` (tiktoken) no boot mesmo sem uso. Lazy init dentro do primeiro `get`/`set`.

### 4.9 Médio — `ContextVar` com `default=[]` mutável

`tools/athena.py:13` e `tools/rag.py:7` usam `default=[]` (mesmo objeto compartilhado). Trocar por `default=None` e `get([]) or []`.

---

## 5. Código morto e legado

### 5.1 Arquivos na raiz para deletar

Todos já em `.gitignore` e `.dockerignore`, nenhum import em `app/` ou `scripts/`:

| Arquivo | Tamanho | Ação |
|---|---|---|
| `chunking-catarata.py` | 8,7 KB | Deletar (mover para branch `legacy/bootstrap-rag` se quiser preservar). |
| `chunking-normas.py` | 3,5 KB | idem |
| `chunking-rdc.py` | 7,2 KB | idem |
| `chunking-regras.py` | 3,9 KB | idem |
| `criar_index_pinecone.py` | 0,6 KB | idem |
| `test_catarata.py` | 0,7 KB | Deletar (script ad-hoc, não é pytest). |
| `iris.json`, `judge.json`, `logging.json`, `rag.json`, `sql.json` | 80–60 KB | Snapshots N8N legados; nenhum `open(...)` em `app/`. Deletar do disco. |
| `indexer.log` | **141 MB** | Deletar imediatamente. |

### 5.2 `scratch/` (18 arquivos)

Confirmado via grep: nada em `app/` importa de `scratch/`. **Deletar a pasta** ou mover para branch `archive/exploratory`.

### 5.3 `tests/test_caracterizacao_v2.py` — duplicado

Apesar do sufixo `_v2`, é a versão **antiga** (mais verbosa, com `print` de debug, **sem** testes para `extract_laterality`). `test_caracterizacao.py` é a versão consolidada. Deletar `_v2`.

Também: remover `tests/` do `.gitignore` (inconsistente — testes existem mas pasta está ignorada).

### 5.4 Módulos e símbolos não usados

| Caminho | Símbolo | Ação |
|---|---|---|
| `app/tools/transcription.py` | arquivo inteiro | `transcribe_audio_tool` nunca é registrado em `app/agent/tools.py`. WS e voice usam o service diretamente. **Deletar o arquivo.** |
| `app/services/memory.py` | `clear_session_history` (139) | Sem referências. Deletar. |
| `app/services/memory.py` | `close_pool` (161) | Sem referências (nem no lifespan). Deletar **OU plugar** no shutdown. |
| `app/agent/evaluator.py` | `_empty_evaluation = _unavailable_metric` (220) | Alias sem consumidor. Deletar. |
| `app/api/chat.py` | rota `/chat` | Comentário declara “endpoint legado, use /api/v1/iris/chat”. Validar com equipe e remover. |
| `app/api/audio.py` | rota `/audio/upload` | Sem consumidor conhecido (a tool LangChain que usaria o filename está morta). Validar e remover. |

### 5.5 Imports não usados

- `app/agent/iris_orchestrator.py`: `json` (6), `SystemMessage`, `AIMessage` (12), `get_chat_model_openai` (22), `settings` (15) — importados e nunca referenciados.
- `app/agent/specialists/sql_analyst.py`: `unicodedata` (20) — uso migrado para `intent.normalize_text`.

### 5.6 BUG runtime em `app/agent/evaluator.py`

`settings.MODEL_NAME` é usado nas linhas 181 e 214, **mas `settings` não está importado**. Em qualquer caminho que execute esses trechos, levanta `NameError`. Adicionar:

```python
from app.core.config import settings
```

### 5.7 Configurações órfãs

- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL` em `app/core/config.py:46-48`. Projeto migrou para LangSmith. Remover.
- `PINECONE_INDEX_CFM`, `PINECONE_INDEX_POP` em `render.yaml`. Não referenciados em `app/` (eram dos `chunking-*.py`). Remover.
- `MODEL_NAME_SQL = "gpt-5.4-mini"` em `config.py`. Verificar se é usado; se não, remover.
- Defaults `MODEL_NAME = "gpt-5.5"` e `MODEL_CLAUDE = "claude-sonnet-4-6"` são sobrepostos pelo `render.yaml` (`gpt-4o`). Alinhar.

### 5.8 Wrapper sem valor

`app/services/transcription.py` (service) vs `app/tools/transcription.py` (adapter LangChain morto) — deletar o adapter conforme 5.4.

---

## 6. Plano de execução sugerido (ordem)

**Fase 1 — Higiene (1h, baixo risco):**
1. Deletar `indexer.log` + adicionar `*.log` ao `.gitignore`.
2. Deletar arquivos da raiz (`chunking-*.py`, `criar_index_pinecone.py`, `test_catarata.py`, `*.json` exceto whitelist).
3. Deletar `scratch/` (mover para branch se quiser).
4. Deletar `tests/test_caracterizacao_v2.py`.
5. Deletar `app/tools/transcription.py`.
6. Limpar imports não usados em `iris_orchestrator.py` e `sql_analyst.py`.
7. Corrigir bug em `evaluator.py` (importar `settings`).
8. Remover vars `LANGFUSE_*` e `PINECONE_INDEX_CFM/POP`.

**Fase 2 — Robustez de runtime (4–6h):**
9. Criar `app/core/clients.py` (singletons OpenAI/Anthropic/Pinecone/httpx).
10. Adicionar `timeout` + `max_retries` em `services/transcription.py` e `services/llm.py`.
11. Backoff exponencial + jitter em `sql_analyst.py` retry loop.
12. Limitar fallbacks in-memory com `deque(maxlen=...)`.
13. WebSocket com `asyncio.wait_for` + heartbeat em `ws.py`.
14. Plugar `close_pool()` no `lifespan` de `main.py`.

**Fase 3 — Memória e disco (4h):**
15. `RotatingFileHandler` ou stdout em vez de `FileHandler` para `indexer.log`.
16. Streaming de upload + limite de tamanho em `audio.py`/`voice.py`/`ws.py`.
17. TTL automático em `temp_audios/`.
18. Mover indexação histórica do endpoint para job externo (apenas D-1 no Render).
19. Lazy init de `PineconeSemanticCache`.

**Fase 4 — Refatoração estrutural (1–2 dias):**
20. Quebrar `sql_analyst.py` em 3 módulos.
21. Extrair prompts para `app/agent/prompts/*.md`.
22. Criar `app/services/supabase_client.py` (consolida `_get_headers` + clientes).
23. Refatorar `learning.generate_lessons_from_execution` para tabela de regras.
24. Unificar branches stream/non-stream em `iris_chat.py` e `iris_orchestrator.py`.
25. Avaliar remoção das rotas legadas `/chat` e `/audio/upload`.

---

## 7. Estimativas de ganho

- **Disco/repositório:** ~141 MB (indexer.log) + ~25 KB (arquivos raiz) + ~18 arquivos scratch.
- **Memória runtime:** ~30–80 MB economizados com singletons de clientes (estimativa).
- **Latência:** -100–300ms por request com reuso de conexão HTTP/TLS.
- **Confiabilidade:** elimina pelo menos 3 cenários conhecidos de worker travado (Whisper, Athena retry, WS zumbi).
- **Linhas de código:** redução estimada de ~600 linhas (1244→700 em sql_analyst, 300→150 em learning, 400→250 em iris_chat, +remoções).
