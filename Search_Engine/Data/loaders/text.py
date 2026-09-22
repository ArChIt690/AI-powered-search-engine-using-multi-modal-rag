from pathlib import Path

from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter

from Search_Engine.Schema.document import Document, Section

_HEADER_KEYS = [f"h{level}" for level in range(1, 7)]
_HEADERS_TO_SPLIT_ON = [("#" * level, key) for level, key in enumerate(_HEADER_KEYS, start=1)]


def load_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    sections = [Section(text=text)] if text else []
    return Document(source=path.resolve().as_posix(), file_type="txt", sections=sections)


def load_markdown(path: Path) -> Document:
    """One section per heading, so chunks never mix content from two headings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Keeps blank lines (paragraph breaks) and ignores "#" lines inside code blocks.
    # A new splitter per file: it keeps state between calls and returns its own internal list.
    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the heading line in the text, so keyword search can match it
    )
    sections = []
    for piece in splitter.split_text(text):
        content = piece.page_content.strip()
        if not content:
            continue
        headings = [piece.metadata[key] for key in _HEADER_KEYS if key in piece.metadata]
        sections.append(Section(text=content, heading=" > ".join(headings) or None))
    return Document(source=path.resolve().as_posix(), file_type="md", sections=sections)
