from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO_TRANSCRIPT = "video_transcript"
    VIDEO_FRAME = "video_frame"


class ChunkMetadata(BaseModel):
    """The typed fields every chunk's `metadata` holds, set by Metadata Enrichment.

    One definition for the Elasticsearch field mapping (1.5) and Metadata Filtering (Part 2), so a field always
    has the same type in every chunk. `modality`, `page` and `timestamp` are top-level Chunk fields, not here.
    """

    model_config = {"extra": "forbid"}  # a misspelled field fails loudly instead of creating a new ES field

    doc_id: str = Field(description="Stable id of the source file (hash of its path); the same file always gets the same id.")
    chunk_index: int = Field(description="Position of the chunk in its document: text chunks first, then pictures.")
    file_name: str = Field(description="File name with extension, e.g. 'report.pdf'.")
    file_type: str = Field(description="Short file type without the dot: pdf, md, csv, png, mp4...")
    content: Literal["text", "table", "record", "image", "chart", "frame"] = Field(
        description="What the chunk holds: prose text, a table, CSV/JSON/XML records, or an image, chart or video frame.",
    )
    title: str = Field(description="PDF title when the file has one, otherwise the file name without its extension.")
    section: str | None = Field(default=None, description="Markdown heading path, e.g. 'Setup > Usage'.")
    author: str | None = Field(default=None, description="Author from the file's metadata (PDFs).")
    created: str | None = Field(default=None, description="ISO date the file says it was created (PDF, photo EXIF, video).")
    file_modified: str | None = Field(default=None, description="ISO UTC time the file was last modified on disk.")
    ingested_at: str = Field(description="ISO UTC time the chunk was enriched.")
    page_count: int | None = Field(default=None, description="Number of pages (PDFs).")
    duration_s: float | None = Field(default=None, description="Length in seconds (videos).")
    language: str | None = Field(default=None, description="Spoken language detected by Whisper (videos).")


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
        description="Kind of content: text, image, video transcript or video frame.",
    )
    page: int | None = Field(
        default=None,
        description="1-based page number for PDFs; None for other file types.",
    )
    timestamp: float | None = Field(
        default=None,
        description="Seconds into the video where this chunk starts; None for non-video content.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="ChunkMetadata fields (file name/type, content, section, dates...), set by Chunking, Image Embeddings and Metadata Enrichment. Used by Metadata Filtering.",
    )
    text_embedding: list[float] | None = Field(
        default=None,
        description="Unit-length text embedding (TEXT_EMBEDDING_DIM floats), used by Semantic Search.",
    )
    image_embedding: list[float] | None = Field(
        default=None,
        description="Unit-length CLIP image embedding (IMAGE_EMBEDDING_DIM floats), for pictures, PDF charts and video frames.",
    )
