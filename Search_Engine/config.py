from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_index: str = "search_chunks"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Embeddings
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    text_embedding_dim: int = 384
    text_query_instruction: str = "Represent this sentence for searching relevant passages: "  # BGE query prefix
    image_embedding_dim: int = 512  # CLIP ViT-B/32, filled in Phase 3
    embedding_batch_size: int = 32

    # Chunking
    # auto = smart for structured files (CSV/JSON/XML), semantic for prose (TXT/MD/PDF)
    chunking_strategy: Literal["auto", "smart", "semantic"] = "auto"
    chunk_size: int = 800  # max characters per chunk
    chunk_overlap: int = 100  # characters carried over between neighbouring chunks
    semantic_breakpoint_percentile: float = 90.0  # higher = fewer, larger semantic chunks
    semantic_min_chunk_size: int = 200  # characters; avoids tiny one-sentence semantic chunks

    # Retrieval
    top_k: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
