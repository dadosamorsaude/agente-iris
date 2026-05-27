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

## 🌐 Deploy no Render

Este repositório contém um arquivo `render.yaml`. Para subir no Render:
1. Conecte seu repositório GitHub ao Render Dashboard.
2. Selecione a opção **Blueprints**.
3. Preencha as variáveis de ambiente solicitadas.

## 🛡️ Segurança

Para acessar a API, é necessário enviar o header:
- `X-API-Key`: Sua chave definida na variável `API_KEY`.
