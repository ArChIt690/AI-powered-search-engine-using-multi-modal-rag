"""VECTOR DATABASE: the Elasticsearch index that Hybrid Search (Phase 2) queries.

One index holds everything: `text` for Keyword Search (BM25), `text_embedding` for Semantic Search,
`image_embedding` for CLIP vectors (Phase 3), and the enriched metadata for Metadata Filtering.
"""

from elasticsearch import Elasticsearch, helpers

from Search_Engine.Schema.chunk import Chunk


def index_mapping(text_dim: int, image_dim: int) -> dict:
    return {
        "properties": {
            "text": {"type": "text", "analyzer": "english"},
            "source": {"type": "keyword"},
            "modality": {"type": "keyword"},
            "page": {"type": "integer"},
            "timestamp": {"type": "float"},
            "metadata": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "section": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                    "author": {"type": "keyword"},
                    "page_count": {"type": "integer"},
                    "file_modified": {"type": "date"},
                    "ingested_at": {"type": "date"},
                    "chunk_index": {"type": "integer"},
                    "chunk_count": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "word_count": {"type": "integer"},
                }
            },
            "text_embedding": {"type": "dense_vector", "dims": text_dim, "index": True, "similarity": "cosine"},
            "image_embedding": {"type": "dense_vector", "dims": image_dim, "index": True, "similarity": "cosine"},
        }
    }


def ensure_index(es: Elasticsearch, index: str, *, text_dim: int, image_dim: int) -> None:
    """Create the index if missing. If it exists, check its vector size matches the embedding model."""
    if not es.indices.exists(index=index):
        es.indices.create(index=index, mappings=index_mapping(text_dim, image_dim))
        return

    properties = es.indices.get_mapping(index=index)[index]["mappings"].get("properties", {})
    existing_dim = properties.get("text_embedding", {}).get("dims")
    if existing_dim != text_dim:
        raise RuntimeError(
            f"Index {index!r} stores {existing_dim}-dim text embeddings but the model produces {text_dim}. "
            f"Delete the index (or use a new ES_INDEX) and ingest again."
        )


def replace_document_chunks(es: Elasticsearch, index: str, source: str, chunks: list[Chunk]) -> int:
    """Delete the file's old chunks, then bulk-insert the new ones, so re-ingesting never leaves stale chunks.

    New chunks become searchable after the next index refresh (automatic within ~1s, or call es.indices.refresh).
    """
    es.delete_by_query(index=index, query={"term": {"source": source}}, refresh=True, conflicts="proceed")
    actions = (
        {"_index": index, "_id": chunk.id, "_source": chunk.model_dump(mode="json", exclude={"id"}, exclude_none=True)}
        for chunk in chunks
    )
    indexed, _ = helpers.bulk(es, actions)
    return indexed
