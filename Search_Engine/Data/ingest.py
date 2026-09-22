"""INGESTION PIPELINE, text path:

Text Extraction -> Chunking (Smart, Semantic) -> Text Embeddings -> Metadata Enrichment -> Vector Database
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from elasticsearch import Elasticsearch

from Search_Engine.config import Settings, get_settings
from Search_Engine.Data.chunking import chunk_document
from Search_Engine.Data.embed import TextEmbedder
from Search_Engine.Data.enrichment import enrich_chunks
from Search_Engine.Data.index import ensure_index, replace_document_chunks
from Search_Engine.Data.loaders import SUPPORTED_EXTENSIONS, load_file

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        es: Elasticsearch | None = None,
        embedder: TextEmbedder | None = None,
    ):
        self.settings = settings or get_settings()
        self.es = es or Elasticsearch(self.settings.es_url)
        self.embedder = embedder or TextEmbedder(
            self.settings.text_embedding_model, self.settings.embedding_batch_size
        )

    def ingest_path(self, path: Path) -> IngestReport:
        """Ingest one file, or every supported file under a folder (recursively)."""
        if not path.exists():
            raise FileNotFoundError(path)
        self._prepare_index()

        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        report = IngestReport()
        for file in files:
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.skipped.append(str(file))
                continue
            try:
                count = self.ingest_file(file)
            except Exception as exc:  # one bad file must not stop the whole run
                logger.exception("Failed to ingest %s", file)
                report.failed[str(file)] = f"{type(exc).__name__}: {exc}"
                continue
            report.files += 1
            report.chunks += count
            logger.info("Ingested %s (%d chunks)", file, count)

        self.es.indices.refresh(index=self.settings.es_index)  # make everything searchable now, once
        return report

    def ingest_file(self, path: Path) -> int:
        s = self.settings
        doc = load_file(path)  # Text Extraction
        chunks = chunk_document(  # Chunking (Smart, Semantic)
            doc,
            strategy=s.chunking_strategy,
            chunk_size=s.chunk_size,
            overlap=s.chunk_overlap,
            embed_fn=self.embedder.embed,
            breakpoint_percentile=s.semantic_breakpoint_percentile,
        )
        self.embedder.embed_chunks(chunks)  # Text Embeddings
        enrich_chunks(chunks, doc)  # Metadata Enrichment
        return replace_document_chunks(self.es, s.es_index, doc.source, chunks)  # Vector Database

    def _prepare_index(self) -> None:
        model_dim = self.embedder.dim
        if model_dim != self.settings.text_embedding_dim:
            raise ValueError(
                f"{self.settings.text_embedding_model} produces {model_dim}-dim vectors, "
                f"but TEXT_EMBEDDING_DIM is {self.settings.text_embedding_dim}"
            )
        ensure_index(
            self.es,
            self.settings.es_index,
            text_dim=model_dim,
            image_dim=self.settings.image_embedding_dim,
        )
