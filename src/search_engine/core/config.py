from functools import lru_cache
from typing import Literal

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

    # Retrieval
    top_k: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
