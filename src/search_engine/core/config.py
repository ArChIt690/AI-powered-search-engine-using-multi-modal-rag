from functools import lru_cache
from typing import Any, Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_text_index: str = "search_chunks_text"  # text, table, record and transcript chunks (bge vectors)
    es_image_index: str = "search_chunks_images"  # image, chart and video-frame chunks (CLIP vectors)

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    index_version_key: str = "search:index_version"  # bumped after every ingest; the Cache layer invalidates on it
    # Embeddings
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    text_embedding_dim: int = 384
    text_query_instruction: str = "Represent this sentence for searching relevant passages: "  # BGE query prefix
    clip_model: str = "clip-ViT-B-32"  # Image Embeddings (CLIP), via sentence-transformers
    image_embedding_dim: int = 512  # CLIP ViT-B/32
    embedding_batch_size: int = 32

    # PDF separation of images, charts and tables
    pdf_min_image_px: int = 64  # skip smaller raster images (icons, bullets, logos)
    pdf_min_chart_pt: float = 72.0  # a drawing cluster must be at least this wide and tall (72 pt = 1 inch)
    pdf_min_chart_paths: int = 3  # and contain this many vector paths, so single boxes and rules are ignored
    pdf_chart_dpi: int = 150  # resolution for rendering charts to PNG

    # Video: Extract Audio & Convert to Text (faster-whisper)
    whisper_model: str = "base"  # tiny, base, small, medium, large-v3...; bigger = better and slower
    whisper_device: str = "auto"  # auto, cpu or cuda
    whisper_compute_type: str = "int8"  # int8 is fast on CPU; float16 on a GPU

    # Video: Takes pictures frame by frame temporarily
    video_frame_interval_s: float = 5.0  # at most one frame per this many seconds
    video_frame_min_change: float = 0.05  # skip frames that differ less than this (0-1) from the last kept one
    video_frame_max_side_px: int = 1024  # frames are downscaled to fit; CLIP uses 224 px anyway

    # Chunking
    # auto = smart for structured files (CSV/JSON/XML), semantic for prose (TXT/MD/PDF)
    chunking_strategy: Literal["auto", "smart", "semantic"] = "auto"
    chunk_size: int = 800  # max characters per chunk
    chunk_overlap: int = 100  # characters carried over between neighbouring chunks
    semantic_breakpoint_percentile: float = 90.0  # higher = fewer, larger semantic chunks
    semantic_min_chunk_size: int = 200  # characters; avoids tiny one-sentence semantic chunks

    # LLMs: two free providers used together, each falling back to the other (see infra/llm_client.py)
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"  # the answering LLM
    groq_fast_model: str = "openai/gpt-oss-20b"  # Query Enhancement, Guardrail; Eval's fallback
    gemini_api_key: SecretStr | None = None
    groq_guard_model: str = "openai/gpt-oss-safeguard-20b"  # Guardrail: judges content against a written policy
    gemini_model: str = "gemini-2.5-flash"  # Eval; the fallback for everything else
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 1  # then the other provider is tried

    # LLM Architecture
    agent_max_tool_calls: int = 4  # per question, then the LLM must answer with what it has
    agent_tool_top_k: int = 5  # passages one `search_documents` call adds
    mcp_servers: dict[str, dict[str, Any]] = {}  # MCP_SERVERS='{"name": {"transport": "stdio", "command": ..., "args": [...]}}'
    eval_pass_score: int = 4  # every Eval score (1-5) must reach this for the answer to be cached
    eval_results_dir: str = "data/eval_results"  # stored separately; never ingested

    # API
    landing_dir: str = "data/landing"  # where uploaded files are kept; their path is the citation source

    # Retrieval
    top_k: int = 10  # reranked chunks given to the LLM
    retrieval_candidates: int = 30  # hits per list (text hybrid, image kNN) fetched for Reranking
    session_history_turns: int = 3  # past question/answer pairs shown to Query Enhancement and the LLM

    # Reranking
    # Cross encoder, local. MiniLM-L6 matched or beat BAAI/bge-reranker-base on eval/ at ~1/3 of the time and
    # ~1/14 of the memory (0.95 s vs 3.3 s per question, ~0.1 GB vs 1.4 GB on this CPU).
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_image_slots: int = 2  # best fused image hits kept, after the text hits (the cross encoder reads text only)

    # FAISS Semantic Cache ("similar question")
    semantic_cache_threshold: float = 0.92  # cosine between the rewritten questions
    semantic_cache_dir: str = "data/faiss_cache"
    semantic_cache_max_entries: int = 10_000  # cleared when full

    # Redis Prompt Caching ("exact or similar prompt") and Sessional Queries
    prompt_cache_ttl_s: int = 24 * 3600
    prompt_cache_distance: float = 0.15  # cosine distance for "similar prompt" (same retrieved chunks only)
    session_ttl_s: int = 3600  # renewed on each question


@lru_cache
def get_settings() -> Settings:
    return Settings()
