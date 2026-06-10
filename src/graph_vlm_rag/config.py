"""Configuration via environment variables"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for graph_vlm_rag. All values come from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_password: str = "password"
    neo4j_url: str = "bolt://localhost:7687"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_vl_model: str = "qwen2.5vl:latest"

    # Docling
    docling_url: str = "http://localhost:5001"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Embeddings (using Ollama's nomic-embed-text)
    embed_model: str = "nomic-embed-text:latest"

    # Domain schema
    domain_schema_path: str = "data/domain_schema.yaml"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()