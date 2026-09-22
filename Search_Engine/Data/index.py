"""VECTOR DATABASE: LangChain's ElasticsearchStore over one Elasticsearch index.

Each stored document is {"text", "text_embedding", "metadata": {...}}: `text` for Keyword Search (BM25),
`text_embedding` for Semantic Search, and the enriched metadata for Metadata Filtering (Phase 2).
The same store is what Elasticsearch Hybrid Search queries in Phase 2.
"""

from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore

from Search_Engine.Schema.chunk import Chunk

_TEXT_WITH_KEYWORD = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}

# Typed fields for Metadata Filtering; anything else in chunk.metadata is mapped dynamically.
METADATA_MAPPINGS = {
    "source": {"type": "keyword"},
    "modality": {"type": "keyword"},
    "page": {"type": "integer"},
    "timestamp": {"type": "float"},
    "doc_id": {"type": "keyword"},
    "file_name": {"type": "keyword"},
    "file_type": {"type": "keyword"},
    "section": _TEXT_WITH_KEYWORD,
    "title": _TEXT_WITH_KEYWORD,
    "author": {"type": "keyword"},
    "page_count": {"type": "integer"},
    "file_modified": {"type": "date"},
    "ingested_at": {"type": "date"},
    "chunk_index": {"type": "integer"},
    "chunk_count": {"type": "integer"},
    "char_count": {"type": "integer"},
    "word_count": {"type": "integer"},
}

# English stemming for BM25 ("running" matches "run").
_INDEX_SETTINGS = {"analysis": {"analyzer": {"default": {"type": "english"}}}}


class VectorDatabase:
    def __init__(self, es: Elasticsearch, index: str, embeddings: Embeddings, text_dim: int):
        self.es = es
        self.index = index
        self.text_dim = text_dim
        self.store = ElasticsearchStore(
            index,
            embedding=embeddings,
            client=es,
            query_field="text",
            vector_query_field="text_embedding",
            # Hybrid = BM25 + kNN in one query. rrf=False: Elasticsearch's built-in RRF needs a paid license,
            # and Rank Fusion is its own step (Reranking) in the architecture anyway.
            strategy=DenseVectorStrategy(hybrid=True, rrf=False),
            num_dimensions=text_dim,
            metadata_mappings=METADATA_MAPPINGS,
            custom_index_settings=_INDEX_SETTINGS,
        )

    def check_dimensions(self) -> None:
        """An existing index must store vectors of the same size the embedding model produces."""
        if not self.es.indices.exists(index=self.index):
            return  # the store creates it on the first insert
        properties = self.es.indices.get_mapping(index=self.index)[self.index]["mappings"].get("properties", {})
        existing_dim = properties.get("text_embedding", {}).get("dims")
        if existing_dim != self.text_dim:
            raise RuntimeError(
                f"Index {self.index!r} stores {existing_dim}-dim text embeddings but the model produces "
                f"{self.text_dim}. Delete the index (or use a new ES_INDEX) and ingest again."
            )

    def replace_document_chunks(self, source: str, chunks: list[Chunk]) -> int:
        """Delete the file's old chunks, then add the new ones, so re-ingesting never leaves stale chunks.

        New chunks become searchable after the next refresh (automatic within ~1s, or call refresh()).
        """
        if self.es.indices.exists(index=self.index):
            self.es.delete_by_query(
                index=self.index, query={"term": {"metadata.source": source}}, refresh=True, conflicts="proceed"
            )
        if not chunks:
            return 0

        ids = self.store.add_embeddings(
            text_embeddings=[(chunk.text, chunk.text_embedding) for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
            refresh_indices=False,
        )
        return len(ids)

    def refresh(self) -> None:
        if self.es.indices.exists(index=self.index):
            self.es.indices.refresh(index=self.index)


def _metadata(chunk: Chunk) -> dict:
    """The store keeps only text, vector and metadata, so the chunk's own fields move into metadata."""
    fields = chunk.model_dump(mode="json", include={"source", "modality", "page", "timestamp"}, exclude_none=True)
    return {**chunk.metadata, **fields}
