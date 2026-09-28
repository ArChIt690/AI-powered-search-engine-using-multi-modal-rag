"""ELASTICSEARCH HYBRID SEARCH against real Elasticsearch (docker compose up -d); skipped when it isn't running.

Stand-in query embedders map words to fixed axes, so each test controls what "similar meaning" is.
"""

import uuid

import pytest
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings

from search_engine.core.config import Settings, get_settings
from search_engine.infra.redis import get_redis
from search_engine.ingestion import vector_store
from search_engine.ingestion.enrichment import enrich
from search_engine.ingestion.vector_store import VectorStore
from search_engine.retrieval.hybrid_search import HYBRID, HybridSearch
from search_engine.schemas.chunk import Chunk, Modality
from search_engine.schemas.document import Document
from search_engine.schemas.query import SearchFilters

pytestmark = pytest.mark.integration


def _es_up() -> bool:
    try:
        return Elasticsearch(get_settings().es_url, request_timeout=2).ping()
    except Exception:
        return False


if not _es_up():
    pytest.skip("Elasticsearch is not running (docker compose up -d)", allow_module_level=True)

# axis 0 = concurrency, axis 1 = baking, axis 2 = unrelated to every chunk
_AXES = {"concurrency": 0, "gopher": 0, "go": 0, "oven": 1, "dessert": 1}


class _AxisEmbeddings(Embeddings):
    def __init__(self, dims: int):
        self.dims = dims

    def embed_query(self, text: str) -> list[float]:
        axis = next((a for word, a in _AXES.items() if word in text.lower().split()), 2)
        return _axis(self.dims, axis)

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def _axis(dims: int, axis: int) -> list[float]:
    vector = [0.0] * dims
    vector[axis] = 1.0
    return vector


@pytest.fixture
def search():
    suffix = uuid.uuid4().hex[:8]
    s = Settings(
        es_text_index=f"test_text_{suffix}", es_image_index=f"test_images_{suffix}",
        index_version_key=f"test:version:{suffix}", retrieval_candidates=5,
    )
    text_store = vector_store._store(s.es_text_index, "text_embedding", 384, _AxisEmbeddings(384), HYBRID)
    image_store = vector_store._store(s.es_image_index, "image_embedding", 512, _AxisEmbeddings(512))
    doc = Document(source="/docs/guide.pdf", file_type="pdf", metadata={"title": "Guide"})
    chunks = [
        Chunk(text="Goroutines are lightweight threads in Go.", source=doc.source, page=1,
              metadata={"content": "text"}, text_embedding=_axis(384, 0)),
        Chunk(text="Bake the cake for forty minutes.", source=doc.source, page=2,
              metadata={"content": "text"}, text_embedding=_axis(384, 1)),
        Chunk(text="chart from guide.pdf, page 3", source=doc.source, modality=Modality.IMAGE, page=3,
              metadata={"content": "chart"}, image_embedding=_axis(512, 0)),
    ]
    VectorStore(s, text_store=text_store, image_store=image_store).write(enrich(doc, chunks))
    yield HybridSearch(s, text_store=text_store, image_store=image_store)
    text_store.client.indices.delete(index=f"{s.es_text_index},{s.es_image_index}", ignore_unavailable=True)
    get_redis().delete(s.index_version_key)


def test_keyword_search_finds_a_chunk_its_embedding_does_not(search):
    # "cake" maps to the unrelated axis, so only BM25 can prefer the baking chunk
    results = search.search("cake")

    assert results.text[0].page_content.startswith("Bake")


def test_semantic_search_finds_a_chunk_with_no_shared_words(search):
    # no word of "gopher concurrency" is in the Goroutines chunk; only kNN can find it
    results = search.search("gopher concurrency")

    assert results.text[0].page_content.startswith("Goroutines")


def test_a_text_query_finds_images_and_hits_keep_ids_and_citations(search):
    results = search.search("concurrency chart")

    [chart] = results.images
    assert chart.metadata["content"] == "chart" and chart.metadata["page"] == 3
    assert all(doc.id and doc.metadata["chunk_id"] == doc.id for doc in results.text + results.images)


def test_metadata_filters_apply_to_both_stores(search):
    assert search.search("go", SearchFilters(file_type=["md"])).text == []
    assert search.search("go", SearchFilters(file_type=["md"])).images == []
    text_only = search.search("go", SearchFilters(content=["text"], file_name=["guide.pdf"]))
    assert text_only.images == []  # a content=text filter never searches the image store
    assert {doc.metadata["page"] for doc in text_only.text} == {1, 2}
