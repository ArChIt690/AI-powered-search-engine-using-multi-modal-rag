"""The diagram's inputs (Text Documents, PDFs/XML/CSV/JSON, PICTURES, VIDEO): turn a file into a Document of
sections (text) and pictures (images, PDF charts, video frames)."""

from collections.abc import Callable
from pathlib import Path

from search_engine.core.exceptions import UnsupportedFileTypeError
from search_engine.ingestion.sources.image import IMAGE_EXTENSIONS, load_image
from search_engine.ingestion.sources.pdf import load_pdf
from search_engine.ingestion.sources.structured import load_csv, load_json, load_xml
from search_engine.ingestion.sources.text import load_markdown, load_text
from search_engine.ingestion.sources.video import VIDEO_EXTENSIONS, load_video
from search_engine.schemas.document import Document

LOADERS: dict[str, Callable[[Path], Document]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
    ".xml": load_xml,
    ".csv": load_csv,
    ".json": load_json,
    **dict.fromkeys(IMAGE_EXTENSIONS, load_image),
    **dict.fromkeys(VIDEO_EXTENSIONS, load_video),
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


def load_file(path: Path) -> Document:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(f"Unsupported file type {path.suffix!r}: {path}")
    return loader(path)
