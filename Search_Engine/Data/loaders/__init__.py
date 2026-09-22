"""TEXT EXTRACTION: turn a file into a Document of sections."""

from collections.abc import Callable
from pathlib import Path

from Search_Engine.Data.loaders.pdf import load_pdf
from Search_Engine.Data.loaders.structured import load_csv, load_json, load_xml
from Search_Engine.Data.loaders.text import load_markdown, load_text
from Search_Engine.Schema.document import Document

LOADERS: dict[str, Callable[[Path], Document]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
    ".xml": load_xml,
    ".csv": load_csv,
    ".json": load_json,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


def load_file(path: Path) -> Document:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type {path.suffix!r}: {path}")
    return loader(path)
