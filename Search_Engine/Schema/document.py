from typing import Any

from pydantic import BaseModel, Field

from Search_Engine.Schema.chunk import Modality


class Section(BaseModel):
    """A structural piece of a loaded file: a PDF page, a Markdown section, a CSV row, a JSON record..."""

    text: str
    page: int | None = None
    heading: str | None = None


class Document(BaseModel):
    """Output of Text Extraction: one loaded file, split into sections but not yet chunked."""

    source: str
    file_type: str
    modality: Modality = Modality.TEXT
    structured: bool = False  # True for record-based files (CSV/JSON/XML): each section is one record
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # file-level info, e.g. PDF title/author
