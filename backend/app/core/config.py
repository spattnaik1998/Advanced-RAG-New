from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    OPENAI_API_KEY: str
    FAISS_PERSIST_PATH: str = "./faiss_index"
    METADATA_DB: str = "./metadata.sqlite"
    COHERE_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Compare"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
