# Iris AI Agent 🤖

Agente de IA especializado em análise de prontuários médicos e conformidade com as normas do CFM. Desenvolvido para migrar fluxos complexos do n8n para uma arquitetura robusta em Python/FastAPI.

## 🚀 Principais Funcionalidades

- **Análise de Prontuários**: Consulta automatizada ao Amazon Athena para extração de indicadores de qualidade.
- **RAG (Retrieval-Augmented Generation)**: Integração com Pinecone para consulta de resoluções do CFM e POPs internos.
- **Streaming de Tokens**: Interface interativa com resposta em tempo real.
- **Segurança**: Proteção via API Key (`X-API-Key`) e configuração de CORS para integração com Lovable.
- **Memória Persistente**: Histórico de chat armazenado em PostgreSQL (Supabase) com suporte a sessões via UUID.
- **Observabilidade**: Tracing via Langfuse e logs estruturados com Loguru.

## 🛠️ Tecnologias Utilizadas

- **Backend**: FastAPI, Python 3.12, Uvicorn.
- **LLM**: LangChain / LangGraph (GPT-5.4 , Claude-Sonnet-4.6).
- **Banco de Dados**: Amazon Athena (PyAthena) e PostgreSQL (Psycopg3).
- **Vetorizadores**: Pinecone.
- **DevOps**: Docker, UV (Fast package manager), Render (Blueprint deployment).

## 📦 Como Rodar Localmente

1. **Instalar dependências**:
   ```bash
   pip install uv
   uv sync
   ```

2. **Configurar variáveis**:
   Copie o `.env.example` para `.env` e preencha suas chaves.

3. **Executar**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## 🔗 Endpoints Auxiliares e Relatórios

### 1. `POST /api/v1/iris/report`
Gera o relatório estruturado para análise de cirurgias de catarata a partir do AWS Athena e RAG clínico.

**Request Body:**
```json
{
  "period": "mes_atual",
  "stratification": "todos",
  "laterality": "todos",
  "clinic": "Juazeiro"
}
```
*   `clinic` (opcional): Filtro de substring case-insensitive para clínicas (máx. 200 caracteres).
*   `period` (opcional): `"hoje" | "ontem" | "ultimos_7_dias" | "ultimos_30_dias" | "mes_atual" | "mes_passado" | "todo_historico"`.

**Response:**
Retorna o resumo estatístico geral, resumo filtrado e lista de pacientes contendo os campos de CPF completo formatado, lateralidade e clínica, além do campo `"filters_applied"` contendo o eco de auditoria dos filtros.

---

### 2. `GET /api/v1/iris/clinics`
Retorna a lista distinta de todas as clínicas ativas na base de dados de oftalmologia, ordenadas alfabeticamente (pt-BR, ignorando acentuações). Possui cache interno de 5 minutos.

**Response:**
```json
{
  "success": true,
  "clinics": [
    "Abaetetuba",
    "Aracaju",
    "Belém",
    "São Paulo"
  ]
}
```

## 🌐 Deploy no Render

Este repositório contém um arquivo `render.yaml`. Para subir no Render:
1. Conecte seu repositório GitHub ao Render Dashboard.
2. Selecione a opção **Blueprints**.
3. Preencha as variáveis de ambiente solicitadas.

## 🛡️ Segurança

Para acessar a API, é necessário enviar o header:
- `X-API-Key`: Sua chave definida na variável `API_KEY`.
