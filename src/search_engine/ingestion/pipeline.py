"""The Ingestion pipeline, 1.1 -> 1.5 in the diagram's order, for one file or a folder (`search-engine ingest`)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from search_engine.core.config import Settings, get_settings
from search_engine.ingestion.chunking import chunk_document
from search_engine.ingestion.enrichment import enrich
from search_engine.ingestion.image_embed import ImageEmbedder
from search_engine.ingestion.sources import SUPPORTED_EXTENSIONS, load_file
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.ingestion.vector_store import VectorStore
from search_engine.schemas.chunk import Chunk
from search_engine.schemas.document import Document

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)  # unsupported file types
    failed: dict[str, str] = field(default_factory=dict)  # path -> error


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings or get_settings()
        # One bge instance for semantic Chunking and Text Embeddings (and the text store's query embedder).
        self.text_embedder = text_embedder or TextEmbedder(self.settings)
        self.image_embedder = image_embedder or ImageEmbedder(self.settings)
        self.store = store or VectorStore(self.settings, text_embedder=self.text_embedder)

    def process(self, doc: Document) -> list[Chunk]:
        """Chunking -> Text Embeddings, and PICTURES -> Image Embeddings, then Metadata Enrichment."""
        text_chunks = self.text_embedder.embed_chunks(chunk_document(doc, self.text_embedder, self.settings))
        picture_chunks = self.image_embedder.embed_pictures(doc)
        return enrich(doc, text_chunks + picture_chunks)

    def ingest_file(self, path: Path) -> int:
        """Text Extraction -> ... -> Vector Database for one file. Returns the number of chunks written."""
        return self.store.write(self.process(load_file(path)))

    def ingest_path(self, path: Path) -> IngestReport:
        """One file, or every file under a folder. One bad file is recorded and doesn't stop the rest."""
        report = IngestReport()
        for file in _files(path):
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.skipped.append(str(file))
                continue
            try:
                report.chunks += self.ingest_file(file)
                report.files += 1
                logger.info("Ingested %s", file)
            except Exception as error:  # a corrupt file must not stop a folder ingest
                logger.exception("Failed to ingest %s", file)
                report.failed[str(file)] = f"{type(error).__name__}: {error}"
        return report


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    # Hidden files (.DS_Store, editor temp files) and partial downloads are not documents.
    return sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith((".", "~")))
