
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application configuration
    app_name: str = "Enterprise HR Policy Agentic RAG Copilot"
    app_env: str = "development"

    # LLM provider configuration (NEW)
    llm_provider: str = "ollama"
    ollama_model: str = "qwen:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"

    # OpenAI configuration (optional fallback)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Tavily configuration
    tavily_api_key: str = ""

    # Pinecone configuration
    pinecone_api_key: str = ""
    pinecone_index_name: str = "hr-policy-minilm-384"
    pinecone_namespace: str = "company-hr-kb"

    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG configuration
    top_k: int = 4
    max_retries: int = 1

    # Security
    admin_api_key: str = "change-me-in-production"

    # Database and file paths
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    # Load environment variables
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
