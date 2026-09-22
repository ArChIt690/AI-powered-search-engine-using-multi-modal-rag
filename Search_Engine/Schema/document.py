from typing import Any

from pydantic import BaseModel, Field

from Search_Engine.Schema.chunk import Modality


class Section(BaseModel):
    """A structural piece of a loaded file: a PDF page, a Markdown section, a CSV row, a JSON record..."""

    text: str = Field(
        description="Extracted text of this section. Records are written as 'field: value' lines.",
    )
    page: int | None = Field(
        default=None,
        description="1-based page number for PDFs; None for other file types.",
    )
    heading: str | None = Field(
        default=None,
        description="Markdown heading path this section sits under, e.g. 'Setup > Usage'; None if there is none.",
    )


class Document(BaseModel):
    """Output of Text Extraction: one loaded file, split into sections but not yet chunked."""

    source: str = Field(
        description="Absolute path (POSIX style) of the loaded file.",
    )
    file_type: str = Field(
        description="Short file type without the dot: txt, md, pdf, xml, csv or json.",
    )
    modality: Modality = Field(
        default=Modality.TEXT,
        description="Kind of content in the file; copied onto every chunk made from it.",
    )
    structured: bool = Field(
        default=False,
        description="True for record-based files (CSV/JSON/XML), where each section is one record. 'auto' chunking uses smart chunking for these.",
    )
    sections: list[Section] = Field(
        default_factory=list,
        description="The file's sections in reading order. Chunks never cross a page or heading boundary between sections.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-level information, e.g. PDF title, author and page count. Copied into every chunk's metadata.",
    )
