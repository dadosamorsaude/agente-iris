from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # LLM Settings
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4.1"          # Avaliador (LLM-as-Judge) + Auditoria de Performance
    MODEL_NAME_SQL: str = "gpt-4.1-mini" # Gerador de SQL Athena (tarefa determinística e bem-definida)
    TEMPERATURE: float = 0.0

    ANTHROPIC_API_KEY: str
    MODEL_CLAUDE: str =  "claude-sonnet-4-6"
    TEMPERATURE_CLAUDE: float = 0.4


    # AWS / Athena Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    ATHENA_DATABASE: str        # schema/database name (ex: pdgt_amorsaude_inteligencia)
    ATHENA_S3_STAGING_DIR: str  # s3://bucket/path/ (ex: s3://meu-bucket/athena-results/)

    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_CFM: Optional[str] = None
    PINECONE_INDEX_POP: Optional[str] = None
    PINECONE_INDEX_TRAIN: Optional[str] = None
    PINECONE_INDEX_EXPANSE: Optional[str] = None
    PINECONE_INDEX_CACHE: Optional[str] = None

    # Security
    AGENTE_API_KEY: str
    ALLOWED_ORIGINS: str
    
    # Memory — PostgreSQL (Optional, falls back to in-memory if not set)
    DATABASE_URL: Optional[str] = None  # postgresql://user:password@host:5432/dbname
    DATABASE_API_KEY: Optional[str] = None

    @property
    def supabase_rest_url(self) -> Optional[str]:
        if not self.DATABASE_URL:
            return None
        import re
        match = re.search(r'postgres\.([a-z0-9]+)', self.DATABASE_URL)
        if match:
            return f"https://{match.group(1)}.supabase.co/rest/v1/"
        return None

    @property
    def aws_region_clean(self) -> str:
        return self.AWS_REGION.lstrip(":").strip()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
