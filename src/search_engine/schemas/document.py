from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from search_engine.schemas.chunk import Modality


class SectionKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


class PictureKind(StrEnum):
    IMAGE = "image"  # a raster image: a standalone picture file, a photo inside a PDF
    CHART = "chart"  # vector drawings in a PDF (charts, diagrams), rendered to an image
    FRAME = "frame"  # a frame sampled from a video


class Section(BaseModel):
    """A structural piece of a loaded file: a PDF page, a Markdown section, a CSV row, a JSON record..."""

    text: str = Field(
        description="Extracted text of this section. Records are written as 'field: value' lines, tables as Markdown.",
    )
    kind: SectionKind = Field(
        default=SectionKind.TEXT,
        description="'table' for a table serialized to Markdown (PDF tables), so smart chunking can split it by rows.",
    )
    page: int | None = Field(
        default=None,
        description="1-based page number for PDFs; None for other file types.",
    )
    heading: str | None = Field(
        default=None,
        description="Markdown heading path this section sits under, e.g. 'Setup > Usage'; None if there is none.",
    )
    timestamp: float | None = Field(
        default=None,
        description="Seconds into the video where this transcript section starts; None for non-video content.",
    )


class Picture(BaseModel):
    """An image found in a file, on its way to PICTURES -> Image Embeddings (CLIP)."""

    data: bytes = Field(
        description="Encoded image bytes, in the format given by 'image_format'.",
    )
    image_format: str = Field(
        default="png",
        description="Encoding of 'data': png, jpeg...",
    )
    kind: PictureKind = Field(
        default=PictureKind.IMAGE,
        description="'image' for raster images, 'chart' for rendered vector drawings from a PDF, 'frame' for video frames.",
    )
    page: int | None = Field(
        default=None,
        description="1-based PDF page the picture was found on; None for other file types.",
    )
    timestamp: float | None = Field(
        default=None,
        description="Seconds into the video where this frame was taken; None for non-video pictures.",
    )
    width: int = Field(description="Width in pixels.")
    height: int = Field(description="Height in pixels.")

    @property
    def modality(self) -> Modality:
        """Modality of the chunk made from this picture."""
        return Modality.VIDEO_FRAME if self.kind == PictureKind.FRAME else Modality.IMAGE


class Document(BaseModel):
    """Output of Text Extraction: one loaded file, split into sections but not yet chunked."""

    source: str = Field(
        description="Absolute path (POSIX style) of the loaded file.",
    )
    file_type: str = Field(
        description="Short file type without the dot: txt, md, pdf, xml, csv, json, png, jpg, mp4...",
    )
    modality: Modality = Field(
        default=Modality.TEXT,
        description="Kind of content in the sections; copied onto every chunk made from them. Picture chunks use Picture.modality.",
    )
    structured: bool = Field(
        default=False,
        description="True for record-based files (CSV/JSON/XML), where each section is one record. 'auto' chunking uses smart chunking for these.",
    )
    sections: list[Section] = Field(
        default_factory=list,
        description="The file's sections in reading order. Chunks never cross a page or heading boundary between sections.",
    )
    pictures: list[Picture] = Field(
        default_factory=list,
        description="Images, PDF charts and video frames, sent to PICTURES -> Image Embeddings instead of Chunking.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-level information, e.g. PDF title, author and page count. Copied into every chunk's metadata.",
    )
