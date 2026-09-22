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

    id: str | None = None  # set by Metadata Enrichment
    text: str
    source: str
    modality: Modality = Modality.TEXT
    page: int | None = None
    timestamp: float | None = None  # seconds into a video (Phase 3)
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None  # CLIP (Phase 3)
