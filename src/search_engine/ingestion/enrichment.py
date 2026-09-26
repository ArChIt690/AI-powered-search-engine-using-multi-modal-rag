"""METADATA ENRICHMENT: runs after both embedding boxes, on every chunk, before the Vector Database.

Gives each chunk a stable id and the typed ChunkMetadata fields that Metadata Filtering and citations use.
No LLM calls: everything comes from the file and what the earlier boxes found.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from search_engine.schemas.chunk import Chunk, ChunkMetadata
from search_engine.schemas.document import Document


def enrich(doc: Document, chunks: list[Chunk]) -> list[Chunk]:
    """Metadata Enrichment. `chunks` are the document's text chunks followed by its picture chunks, in order."""
    if not chunks:
        return []

    doc_id = document_id(doc.source)
    name = PurePosixPath(doc.source)
    shared = {
        "doc_id": doc_id,
        "file_name": name.name,
        "file_type": doc.file_type,
        "title": doc.metadata.get("title") or name.stem,
        "author": doc.metadata.get("author"),
        "created": doc.metadata.get("created"),
        "file_modified": _file_modified(doc.source),
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "page_count": doc.metadata.get("page_count"),
        "duration_s": doc.metadata.get("duration_s"),
        "language": doc.metadata.get("language"),
    }
    for index, chunk in enumerate(chunks):
        chunk.id = f"{doc_id}-{index}"
        # Keeps what Chunking and Image Embeddings already set (content, section); validation fixes every type.
        metadata = ChunkMetadata(**shared, **chunk.metadata, chunk_index=index)
        chunk.metadata = metadata.model_dump(exclude_none=True)
    return chunks


def document_id(source: str) -> str:
    """Same file path, same id: re-ingesting a file replaces its chunks instead of adding duplicates."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def _file_modified(source: str) -> str | None:
    try:
        mtime = Path(source).stat().st_mtime
    except OSError:  # e.g. the file was already moved out of the landing folder
        return None
    return datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds")
