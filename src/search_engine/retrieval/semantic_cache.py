"""FAISS SEMANTIC CACHE: "checks if similar question in cache". HIT -> USER, MISS -> Redis Prompt Caching.

Keyed on the bge embedding of the question after QUERY Enhancement, so a follow-up ("and page 3?") is compared as
the full question it was rewritten to. A HIT needs a similar question (cosine >= semantic_cache_threshold) asked
with the same filters. The whole cache belongs to one corpus version: after a new ingest it starts empty, so an
answer is never served from documents that have since changed.
"""

import json
import logging
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from search_engine.core.config import Settings, get_settings
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.retrieval.filters import filters_key
from search_engine.schemas.query import SearchFilters
from search_engine.schemas.response import CachedAnswer

logger = logging.getLogger(__name__)

_VERSION_FILE = "index_version.json"


class SemanticCache:
    def __init__(self, settings: Settings | None = None, *, embedder: TextEmbedder | None = None):
        self.settings = settings or get_settings()
        self.embedder = embedder or TextEmbedder(self.settings)
        self.directory = Path(self.settings.semantic_cache_dir)
        self._lock = threading.Lock()
        self._store: FAISS | None = None
        self._version: int | None = None

    def lookup(self, question: str, filters: SearchFilters, index_version: int) -> CachedAnswer | None:
        """The cached answer of the most similar earlier question with the same filters, or None (a MISS)."""
        try:
            with self._lock:
                store = self._current(index_version)
                if store.index.ntotal == 0:
                    return None
                hits = store.similarity_search_with_score_by_vector(
                    self.embedder.embed_query(question), k=1, filter={"filters_key": filters_key(filters)}, fetch_k=50
                )
        except Exception as error:  # the cache is an optimisation: failing means a MISS, never a failed search
            logger.warning("FAISS Semantic Cache lookup failed, treating as a miss: %s", error)
            return None
        if not hits or hits[0][1] < self.settings.semantic_cache_threshold:
            return None
        doc, score = hits[0]
        logger.info("FAISS Semantic Cache HIT (cosine %.3f) for %r ~ %r", score, question, doc.page_content)
        return CachedAnswer.model_validate_json(doc.metadata["answer"])

    def store(self, question: str, filters: SearchFilters, index_version: int, answer: CachedAnswer) -> None:
        """Caches an answer (RESULT "goes for caching") and saves the cache to disk."""
        try:
            with self._lock:
                store = self._current(index_version)
                if store.index.ntotal >= self.settings.semantic_cache_max_entries:
                    store = self._reset(index_version)
                store.add_embeddings(
                    [(question, self.embedder.embed_query(question))],
                    metadatas=[{
                        "filters_key": filters_key(filters),
                        "answer": answer.model_dump_json(),
                        "created_at": datetime.now(UTC).isoformat(),
                    }],
                )
                store.save_local(str(self.directory))
        except Exception as error:
            logger.warning("Could not write to the FAISS Semantic Cache: %s", error)

    def _current(self, index_version: int) -> FAISS:
        """The cache for this corpus version: in memory, else loaded from disk, else a new empty one."""
        if self._store is not None and self._version == index_version:
            return self._store
        if self._saved_version() == index_version:
            self._store = FAISS.load_local(
                str(self.directory),
                self.embedder,
                allow_dangerous_deserialization=True,  # a pickle only this class writes, in our own data folder
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            self._version = index_version
            return self._store
        return self._reset(index_version)

    def _reset(self, index_version: int) -> FAISS:
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / _VERSION_FILE).write_text(json.dumps({"index_version": index_version}))
        self._store = FAISS(
            embedding_function=self.embedder,
            # bge vectors are unit length, so the inner product is the cosine similarity
            index=faiss.IndexFlatIP(self.settings.text_embedding_dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self._version = index_version
        return self._store

    def _saved_version(self) -> int | None:
        try:
            if not (self.directory / "index.faiss").exists():
                return None
            return json.loads((self.directory / _VERSION_FILE).read_text())["index_version"]
        except (OSError, ValueError, KeyError):
            return None
