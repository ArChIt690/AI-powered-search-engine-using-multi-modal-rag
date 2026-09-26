"""The Vector Database against real Elasticsearch + Redis (docker compose up -d); skipped when they aren't running."""

import uuid

import pytest
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings

from search_engine.core.config import Settings, get_settings
from search_engine.infra.redis import get_redis
from search_engine.ingestion import vector_store
from search_engine.ingestion.enrichment import enrich
from search_engine.ingestion.vector_store import VectorStore
from search_engine.schemas.chunk import Chunk, Modality
from search_engine.schemas.document import Document

pytestmark = pytest.mark.integration


def _es_up() -> bool:
    try:
        return Elasticsearch(get_settings().es_url, request_timeout=2).ping()
    except Exception:
        return False


if not _es_up():
    pytest.skip("Elasticsearch is not running (docker compose up -d)", allow_module_level=True)


class _AxisEmbeddings(Embeddings):
    """Query embedder for the test: 'go' queries point one way, 'cake' queries another."""

    def __init__(self, dims: int):
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        vector[0 if "go" in text.lower() else 1] = 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def _axis(dims: int, axis: int) -> list[float]:
    vector = [0.0] * dims
    vector[axis] = 1.0
    return vector


@pytest.fixture
def store():
    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        es_text_index=f"test_text_{suffix}", es_image_index=f"test_images_{suffix}", index_version_key=f"test:version:{suffix}"
    )
    stores = VectorStore(
        settings,
        text_store=vector_store._store(settings.es_text_index, "text_embedding", 384, _AxisEmbeddings(384)),
        image_store=vector_store._store(settings.es_image_index, "image_embedding", 512, _AxisEmbeddings(512)),
    )
    yield stores
    indexes = f"{settings.es_text_index},{settings.es_image_index}"  # by name: ES refuses wildcard deletes
    stores.text_store.client.indices.delete(index=indexes, ignore_unavailable=True)
    get_redis().delete(settings.index_version_key)


def _document_chunks(n_text: int = 2) -> tuple[Document, list[Chunk]]:
    doc = Document(source="/docs/go_guide.pdf", file_type="pdf", metadata={"title": "Go Guide", "page_count": 3})
    text = [
        Chunk(text="Goroutines are lightweight threads in Go.", source=doc.source, page=1,
              metadata={"content": "text"}, text_embedding=_axis(384, 0)),
        Chunk(text="Bake the cake for forty minutes.", source=doc.source, page=2,
              metadata={"content": "text"}, text_embedding=_axis(384, 1)),
    ][:n_text]
    pictures = [
        Chunk(text="chart from go_guide.pdf, page 3", source=doc.source, modality=Modality.IMAGE, page=3,
              metadata={"content": "chart"}, image_embedding=_axis(512, 0)),
    ]
    return doc, enrich(doc, text + pictures)


def test_chunks_are_written_through_langchain_and_found_again(store):
    doc, chunks = _document_chunks()

    assert store.write(chunks) == 3

    [hit] = store.text_store.similarity_search("how do go goroutines work", k=1)
    assert hit.page_content.startswith("Goroutines")
    assert hit.metadata["page"] == 1 and hit.metadata["title"] == "Go Guide" and hit.metadata["modality"] == "text"
    [picture] = store.image_store.similarity_search("go chart", k=1)
    assert picture.metadata["content"] == "chart" and picture.metadata["modality"] == "image"


def test_metadata_is_typed_in_the_index_so_filters_work(store):
    _, chunks = _document_chunks()
    store.write(chunks)

    hits = store.text_store.similarity_search("anything go", k=5, filter=[{"range": {"metadata.page": {"gte": 2}}}])

    assert [h.metadata["page"] for h in hits] == [2]
    mapping = store.text_store.client.indices.get_mapping(index=store.settings.es_text_index)
    properties = next(iter(mapping.body.values()))["mappings"]["properties"]["metadata"]["properties"]
    assert properties["page"]["type"] == "integer" and properties["created"]["type"] == "date"


def test_reingesting_a_file_replaces_its_chunks_instead_of_duplicating(store):
    _, chunks = _document_chunks(n_text=2)
    store.write(chunks)
    _, fewer = _document_chunks(n_text=1)  # the file changed: one text chunk fewer

    store.write(fewer)

    client = store.text_store.client
    assert client.count(index=store.settings.es_text_index)["count"] == 1
    assert client.count(index=store.settings.es_image_index)["count"] == 1


def test_every_write_bumps_the_index_version(store):
    _, chunks = _document_chunks()

    store.write(chunks)
    store.write(chunks)

    assert get_redis().get(store.settings.index_version_key) == "2"
