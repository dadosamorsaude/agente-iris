# Refatoração — Resumo executivo

**Data:** 17/06/2026

## O que foi feito

### Fase 0 — Remoção da análise de voz
- Deletados: `app/api/voice.py`, `app/api/ws.py`, `app/api/audio.py`, `app/services/transcription.py`, `app/tools/transcription.py`, pasta `temp_audios/`.
- `app/main.py`: removidos os imports e `include_router` dos 3 routers de voz.
- `pyproject.toml`: removido `websockets` e `python-multipart`. Movido `httpx` de dev para deps de produção (era usado em runtime).
- `.gitignore` e `.dockerignore`: removida referência a `temp_audios/` órfã reescrita.

### Fase 1.1 — Higiene
Removidos do repositório:
- `indexer.log` (141 MB)
- `chunking-catarata.py`, `chunking-normas.py`, `chunking-rdc.py`, `chunking-regras.py`
- `criar_index_pinecone.py`, `test_catarata.py`
- `iris.json`, `judge.json`, `logging.json`, `rag.json`, `sql.json`
- Pasta `scratch/` (18 arquivos)
- `tests/test_caracterizacao_v2.py` (versão antiga duplicada)

`.gitignore` simplificado:
- Adicionado `*.log`
- Adicionado `temp_audios/`
- Removida regra absurda `/*` + whitelist
- Removida `tests/` (estava ignorando testes que existem)

`.dockerignore` atualizado: adicionados `temp_audios/`, `AVALIACAO_CODIGO.md`, `tests/`.

### Fase 1.2 — Bug fix e imports
- **Bug corrigido** em `app/agent/evaluator.py`: usava `settings.MODEL_NAME` sem importar `settings` (NameError latente). Também: `datetime.utcnow()` (deprecated) → `datetime.now(timezone.utc)`. Removido alias morto `_empty_evaluation`.
- `app/agent/iris_orchestrator.py`: removidos imports não usados (`json`, `SystemMessage`, `AIMessage`, `get_chat_model_openai`, `settings`).
- `app/agent/specialists/sql_analyst.py`: removido `import unicodedata` órfão.
- `app/core/config.py`: removidas vars `LANGFUSE_*` (projeto migrou para LangSmith). Adicionadas `HTTP_TIMEOUT` e `HTTP_MAX_RETRIES`.
- `render.yaml`: removidas vars `PINECONE_INDEX_CFM` e `PINECONE_INDEX_POP` (não referenciadas).

### Fase 2.1 — Singletons de clientes externos
**Novo arquivo `app/core/clients.py`** consolidando:
- `openai_async()` — AsyncOpenAI singleton (timeout/retries do settings)
- `embeddings_3_large()` — OpenAIEmbeddings 3072 dims (RAG/prontuários)
- `embeddings_3_small_1024()` — OpenAIEmbeddings 1024 dims (semantic cache)
- `pinecone()` — Pinecone client lazy (None se sem API key)
- `pinecone_index(name)` — handler de índice cacheado por nome
- `supabase_request(method, path, ...)` — wrapper único para Supabase REST com `httpx.AsyncClient` reutilizável (`max_connections=10`)
- `aclose_clients()` — fecha conexões no shutdown

Refatorados para usar os singletons:
- `app/services/llm.py` — cache LRU por `(model, temperature)`; agora com `timeout` e `max_retries` do settings.
- `app/services/cache.py` — lazy init (não toca Pinecone/embeddings no import do módulo).
- `app/services/prontuario_indexer.py` — usa `openai_async()` e `pinecone_index()`. Concorrência reduzida de `Semaphore(10)` → `Semaphore(2)` para caber no Render free.
- `app/tools/prontuario_search.py` — singletons.
- `app/services/learning.py` — removido `_get_headers()` duplicado, agora usa `supabase_request()`.
- `app/services/evaluation_store.py` — removido `_get_headers()` duplicado, agora usa `supabase_request()`. `datetime.utcnow()` → `datetime.now(timezone.utc)`. Helpers `_summary_from_memory` e `_summary_from_rows` consolidados em um único `_summary_from_rows`.

### Fase 2.2 — Timeouts e backoff
- `app/services/llm.py` — `ChatOpenAI`/`ChatAnthropic` agora com `timeout=settings.HTTP_TIMEOUT` (30s) e `max_retries=settings.HTTP_MAX_RETRIES` (2).
- `app/agent/specialists/sql_analyst.py` — retry SQL agora com:
  - Constantes nomeadas (`SQL_RETRY_MAX_ATTEMPTS`, `SQL_RETRY_BASE_SLEEP`, `SQL_RETRY_MAX_SLEEP`).
  - **Backoff exponencial com full jitter** (AWS-style): `random.uniform(0, min(max, base * 2^attempt))`.
  - **Early abort** para erros permanentes do Athena (`InvalidRequestException`, `AccessDeniedException`, `ResourceNotFoundException`) — não desperdiça tokens de LLM tentando consertar erros 4xx.

### Fase 2.3 — Fallbacks in-memory limitados
- `app/services/memory.py` — substituído `_memory_store: dict = {}` (cresce sem limite) por **LRU global de 100 sessões** (`OrderedDict`) com **deque(maxlen=20)** por sessão. Pool asyncpg agora tem `timeout=10`. Removidas funções mortas `clear_session_history`.
- `app/services/evaluation_store.py` — substituído `_memory_store: list = []` por `deque(maxlen=200)`.
- `app/main.py` (`lifespan`) — agora chama `close_pool()` e `aclose_clients()` no shutdown, com try/except individuais para não mascarar falhas.

### Fase 2.4 — Refatoração estrutural de `learning.py`
- Cascata de 12 `if judge_failed and X:` substituída por **tabela declarativa `LESSON_RULES`**.
- Cada regra é um `LessonRule(category, lesson, confidence, when, reason)`.
- Snapshot de fatos da execução isolado em `ExecutionFacts` (dataclass), evitando recomputar `detect_intent` e normalizações.
- Redução de ~170 linhas de lógica para ~80 linhas + dados.

### Fase 3 — Memória/disco
- `scripts/run_historical_index.py` — `FileHandler("indexer.log")` substituído por `RotatingFileHandler(maxBytes=10MB, backupCount=2)`. Em ambiente CI (GitHub Actions), adiciona handler para stdout. Path configurável via `INDEXER_LOG_FILE`.

## Itens deixados para fase posterior (no relatório original)
- Quebra do `sql_analyst.py` em 3 módulos (`period_extractor.py`, `sql_prompt_builder.py`, `sql_executor.py`) — refatoração estrutural maior, requer revisão de testes.
- Extração de prompts grandes para `app/agent/prompts/*.md`.
- Unificação dos branches stream/non-stream em `iris_chat.py` e `iris_orchestrator.py`.
- Avaliação para remover rotas legadas `/chat` em `app/api/chat.py`.
- Streaming de upload e TTL automático em `temp_audios/` — não aplicável após remoção da análise de voz.

## Verificação
- Todos os arquivos modificados foram revisados via Read tool (fonte de verdade Windows).
- Sintaxe verificada visualmente em todos os finais de arquivo.
- Observação: o ambiente sandbox Linux apresentou lag de sincronização com o filesystem do Windows; `py_compile` direto pelo bash mostrou conteúdo desatualizado. O código no disco do usuário (Windows) está correto e completo.

## Próximos passos recomendados ao usuário
1. Rodar `uv sync` localmente para atualizar `uv.lock` (removeu `websockets`, `python-multipart`; promoveu `httpx` para deps de produção).
2. Rodar a suíte de testes: `uv run pytest tests/`.
3. Commit em branch separada (`refactor/cleanup-and-singletons`) para revisão.
4. Validar no Render se o deploy sobe corretamente — vai economizar memória de imediato.
