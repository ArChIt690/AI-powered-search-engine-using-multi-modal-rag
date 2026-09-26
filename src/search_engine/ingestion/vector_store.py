"""VECTOR DATABASE: enriched chunks -> Elasticsearch, through LangChain `ElasticsearchStore`s.

Two stores in the same Elasticsearch, because a LangChain store manages one vector field per index:
- text store  (es_text_index):  text, table, record and transcript chunks, bge `text_embedding` (384-dim)
- image store (es_image_index): image, chart and video-frame chunks, CLIP `image_embedding` (512-dim)
LangChain creates each index (vector field, typed metadata, English analyzer) and does every write.
Documents look like {text, text_embedding | image_embedding, metadata: {...}}; the Retrieval pipeline reads them.
"""

import logging
import types
from itertools import groupby
from typing import Any, Literal, Union, get_args, get_origin

from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore
from redis import RedisError

from search_engine.core.config import Settings, get_settings
from search_engine.infra.elasticsearch import get_es_client
from search_engine.infra.redis import get_redis
from search_engine.ingestion.image_embed import ClipTextEmbeddings
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.schemas.chunk import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# English analyzer as the default: the `text` field (mapped by LangChain on first write) gets stemming and
# stop words, so the BM25 half of hybrid search matches "running" to "run".
INDEX_SETTINGS = {"analysis": {"analyzer": {"default": {"type": "english"}}}}

_TEXT_FIELDS = {"title", "section", "author", "file_name"}  # full-text searchable, plus .keyword for exact filters
_DATE_FIELDS = {"created", "file_modified", "ingested_at"}


def metadata_mappings() -> dict[str, dict[str, Any]]:
    """Elasticsearch types for every `metadata.*` field, generated from ChunkMetadata so the two never drift."""
    mappings: dict[str, dict[str, Any]] = {
        "modality": {"type": "keyword"},
        "page": {"type": "integer"},
        "timestamp": {"type": "float"},
    }
    for name, field in ChunkMetadata.model_fields.items():
        if name in _DATE_FIELDS:
            mappings[name] = {"type": "date"}
        elif name in _TEXT_FIELDS:
            mappings[name] = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}}
        else:
            python_type = _base_type(field.annotation)
            mappings[name] = {"type": {int: "integer", float: "float"}.get(python_type, "keyword")}
    return mappings


def document_metadata(chunk: Chunk) -> dict[str, Any]:
    """What a chunk stores under `metadata`: its ChunkMetadata fields plus modality, page and timestamp."""
    extra = {"modality": chunk.modality.value, "page": chunk.page, "timestamp": chunk.timestamp}
    return {**chunk.metadata, **{key: value for key, value in extra.items() if value is not None}}


class VectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        text_embedder: TextEmbedder | None = None,
        text_store: ElasticsearchStore | None = None,
        image_store: ElasticsearchStore | None = None,
    ):
        self.settings = settings or get_settings()
        s = self.settings
        self.text_store = text_store or _store(
            s.es_text_index, "text_embedding", s.text_embedding_dim, text_embedder or TextEmbedder(s)
        )
        self.image_store = image_store or _store(
            s.es_image_index, "image_embedding", s.image_embedding_dim, ClipTextEmbeddings()
        )

    def write(self, chunks: list[Chunk]) -> int:
        """Vector Database. Replaces each document's chunks (old ones deleted first), then bumps the index version.

        `chunks` must be enriched (ids and doc_id set). Returns the number of chunks written.
        """
        by_doc = groupby(sorted(chunks, key=lambda c: c.metadata["doc_id"]), key=lambda c: c.metadata["doc_id"])
        for doc_id, doc_chunks in by_doc:
            doc_chunks = list(doc_chunks)
            self.delete_document(doc_id)
            pictures = [c for c in doc_chunks if c.image_embedding is not None]
            texts = [c for c in doc_chunks if c.image_embedding is None]
            _add(self.text_store, texts, "text_embedding")
            _add(self.image_store, pictures, "image_embedding")
        self._bump_index_version()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Removes a file's chunks from both indexes, so a re-ingested file that shrank leaves nothing stale."""
        indexes = ((self.text_store, self.settings.es_text_index), (self.image_store, self.settings.es_image_index))
        for store, index in indexes:
            store.client.delete_by_query(
                index=index,
                query={"term": {"metadata.doc_id": doc_id}},
                refresh=True,
                conflicts="proceed",
                ignore_unavailable=True,  # the index doesn't exist before the first write
            )

    def _bump_index_version(self) -> None:
        try:
            get_redis().incr(self.settings.index_version_key)
        except RedisError:
            # The chunks are already safely in Elasticsearch; cached answers then expire by TTL instead.
            logger.warning("Could not bump %s in Redis; cached answers expire by TTL only", self.settings.index_version_key)


def _store(index: str, vector_field: str, dims: int, embedding) -> ElasticsearchStore:
    return ElasticsearchStore(
        index_name=index,
        client=get_es_client(),
        embedding=embedding,  # embeds search queries (Part 2); ingestion passes precomputed vectors
        strategy=DenseVectorStrategy(),
        vector_query_field=vector_field,
        query_field="text",
        num_dimensions=dims,
        metadata_mappings=metadata_mappings(),
        custom_index_settings=INDEX_SETTINGS,
    )


def _add(store: ElasticsearchStore, chunks: list[Chunk], vector_field: str) -> None:
    if not chunks:
        return
    missing = [c.id for c in chunks if getattr(c, vector_field) is None]
    if missing:
        raise ValueError(f"{len(missing)} chunks have no {vector_field}, e.g. {missing[0]}: embed them before writing")
    store.add_embeddings(
        text_embeddings=[(c.text, getattr(c, vector_field)) for c in chunks],
        metadatas=[document_metadata(c) for c in chunks],
        ids=[c.id for c in chunks],
    )


def _base_type(annotation: Any) -> Any:
    """`str | None` -> str, `Literal[...]` -> Literal, `int` -> int."""
    if get_origin(annotation) in (Union, types.UnionType):
        annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
    return Literal if get_origin(annotation) is Literal else annotation
