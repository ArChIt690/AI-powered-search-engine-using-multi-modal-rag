import re
from pathlib import Path

import pymupdf

from Search_Engine.Schema.document import Document, Section

_TEXT_BLOCK = 0  # PyMuPDF block type: 0 = text, 1 = image


def load_pdf(path: Path) -> Document:
    """Text only, one section per page.

    Phase 3 extends this loader to separate images, charts and tables (images/charts go to PICTURES).
    """
    sections = []
    with pymupdf.open(path) as pdf:
        metadata = {key: value for key in ("title", "author") if (value := (pdf.metadata or {}).get(key))}
        metadata["page_count"] = pdf.page_count

        for number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [_clean(block[4]) for block in blocks if block[6] == _TEXT_BLOCK]
            text = "\n\n".join(p for p in paragraphs if p)
            if text:
                sections.append(Section(text=text, page=number))

    return Document(source=path.resolve().as_posix(), file_type="pdf", sections=sections, metadata=metadata)


def _clean(block_text: str) -> str:
    """A PDF block is one paragraph broken into visual lines: re-join the lines and undo hyphenation."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", block_text)
    return re.sub(r"\s+", " ", text).strip()
