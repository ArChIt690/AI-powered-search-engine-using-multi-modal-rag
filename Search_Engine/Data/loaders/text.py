import re
from pathlib import Path

from Search_Engine.Schema.document import Document, Section

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def load_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    sections = [Section(text=text)] if text else []
    return Document(source=path.resolve().as_posix(), file_type="txt", sections=sections)


def load_markdown(path: Path) -> Document:
    """One section per heading, so chunks never mix content from two headings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    headings = list(_MD_HEADING.finditer(text))

    sections = []
    preamble = text[: headings[0].start()] if headings else text
    if preamble.strip():
        sections.append(Section(text=preamble.strip()))

    ends = [match.start() for match in headings[1:]] + [len(text)]
    for match, end in zip(headings, ends):
        heading = match.group(1)
        body = text[match.end() : end].strip()
        # Keep the heading in the text too, so keyword search can match it.
        sections.append(Section(text=f"{heading}\n\n{body}" if body else heading, heading=heading))

    return Document(source=path.resolve().as_posix(), file_type="md", sections=sections)
