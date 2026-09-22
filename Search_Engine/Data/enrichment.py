"""METADATA ENRICHMENT: runs after embedding, before the Vector Database.

Adds the fields that Metadata Filtering (Phase 2) filters on.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from Search_Engine.Schema.chunk import Chunk
from Search_Engine.Schema.document import Document


def document_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def enrich_chunks(chunks: list[Chunk], doc: Document) -> list[Chunk]:
    path = Path(doc.source)
    doc_id = document_id(doc.source)
    file_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat() if path.exists() else None
    ingested_at = datetime.now(UTC).isoformat()

    for index, chunk in enumerate(chunks):
        # Deterministic id: re-ingesting the same file produces the same ids.
        chunk.id = f"{doc_id}-{index:05d}"
        chunk.metadata.update(doc.metadata)
        chunk.metadata.update(
            doc_id=doc_id,
            file_name=path.name,
            file_type=doc.file_type,
            file_modified=file_modified,
            ingested_at=ingested_at,
            chunk_index=index,
            chunk_count=len(chunks),
            char_count=len(chunk.text),
            word_count=len(chunk.text.split()),
        )
    return chunks
