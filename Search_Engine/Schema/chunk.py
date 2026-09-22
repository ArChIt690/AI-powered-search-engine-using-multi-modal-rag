from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO_TRANSCRIPT = "video_transcript"
    VIDEO_FRAME = "video_frame"
    EVAL = "eval"


class Chunk(BaseModel):
    """The unit that flows through ingestion and is stored in the Vector Database."""

    id: str | None = Field(
        default=None,
        description="Unique chunk id '<doc_id>-<chunk_index>', set by Metadata Enrichment. Used as the Elasticsearch _id.",
    )
    text: str = Field(
        description="The chunk's content: what gets embedded, keyword-searched (BM25) and passed to the LLM as context.",
    )
    source: str = Field(
        description="Absolute path (POSIX style) of the file this chunk came from.",
    )
    modality: Modality = Field(
        default=Modality.TEXT,
        description="Kind of content: text, image, video transcript, video frame or eval result.",
    )
    page: int | None = Field(
        default=None,
        description="1-based page number for PDFs; None for other file types.",
    )
    timestamp: float | None = Field(
        default=None,
        description="Seconds into the video where this chunk starts (Phase 3); None for non-video content.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields from Chunking and Metadata Enrichment (file name/type, section, dates, counts...), used by Metadata Filtering.",
    )
    text_embedding: list[float] | None = Field(
        default=None,
        description="Unit-length text embedding (TEXT_EMBEDDING_DIM floats), used by Semantic Search.",
    )
    image_embedding: list[float] | None = Field(
        default=None,
        description="CLIP image embedding (IMAGE_EMBEDDING_DIM floats), filled in Phase 3.",
    )
