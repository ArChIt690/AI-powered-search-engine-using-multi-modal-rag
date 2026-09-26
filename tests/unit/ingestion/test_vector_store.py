import pytest

from search_engine.core.config import Settings
from search_engine.ingestion import vector_store
from search_engine.ingestion.vector_store import VectorStore, document_metadata, metadata_mappings
from search_engine.schemas.chunk import Chunk, ChunkMetadata, Modality


class _FakeClient:
    def __init__(self, log):
        self.log = log

    def delete_by_query(self, index, query, **kwargs):
        self.log.append(("delete", index, query["term"]["metadata.doc_id"]))


class _FakeStore:
    """Records the LangChain calls instead of talking to Elasticsearch."""

    def __init__(self, name, log):
        self.name, self.log, self.client = name, log, _FakeClient(log)

    def add_embeddings(self, text_embeddings, metadatas, ids):
        self.log.append(("add", self.name, ids, [vector for _, vector in text_embeddings], metadatas))


class _FakeRedis:
    def __init__(self):
        self.counters = {}

    def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1


@pytest.fixture
def setup(monkeypatch):
    log, redis = [], _FakeRedis()
    monkeypatch.setattr(vector_store, "get_redis", lambda: redis)
    settings = Settings(es_text_index="t_idx", es_image_index="i_idx")
    store = VectorStore(settings, text_store=_FakeStore("text", log), image_store=_FakeStore("image", log))
    return store, log, redis


def _meta(doc_id: str, index: int, content: str) -> dict:
    return {"doc_id": doc_id, "chunk_index": index, "content": content, "file_name": "a.pdf", "file_type": "pdf",
            "title": "a", "ingested_at": "2026-09-26T00:00:00+00:00"}


def _chunks(doc_id: str = "d1") -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}-0", text="Intro.", source="/a.pdf", page=1, metadata=_meta(doc_id, 0, "text"), text_embedding=[0.1]),
        Chunk(id=f"{doc_id}-1", text="chart from a.pdf, page 2", source="/a.pdf", modality=Modality.IMAGE, page=2,
              metadata=_meta(doc_id, 1, "chart"), image_embedding=[0.9]),
    ]


def test_text_and_picture_chunks_go_to_their_own_store_after_old_ones_are_deleted(setup):
    store, log, redis = setup

    written = store.write(_chunks())

    assert written == 2
    assert log[:2] == [("delete", "t_idx", "d1"), ("delete", "i_idx", "d1")]  # replace, not duplicate
    (_, text_name, text_ids, text_vectors, _), (_, image_name, image_ids, image_vectors, image_meta) = log[2:]
    assert (text_name, text_ids, text_vectors) == ("text", ["d1-0"], [[0.1]])
    assert (image_name, image_ids, image_vectors) == ("image", ["d1-1"], [[0.9]])
    assert image_meta[0]["modality"] == "image" and image_meta[0]["page"] == 2
    assert redis.counters == {"search:index_version": 1}


def test_each_document_in_a_batch_is_replaced_separately(setup):
    store, log, redis = setup

    store.write(_chunks("d1") + _chunks("d2"))

    deletes = [entry[1:] for entry in log if entry[0] == "delete"]
    assert deletes == [("t_idx", "d1"), ("i_idx", "d1"), ("t_idx", "d2"), ("i_idx", "d2")]
    assert redis.counters["search:index_version"] == 1  # one bump per write call


def test_chunk_without_its_vector_is_refused(setup):
    store, _, _ = setup
    chunk = _chunks()[0].model_copy(update={"text_embedding": None})

    with pytest.raises(ValueError, match="no text_embedding"):
        store.write([chunk])


def test_redis_down_does_not_fail_the_write(monkeypatch, setup, caplog):
    store, log, _ = setup

    def broken():
        raise vector_store.RedisError("connection refused")

    monkeypatch.setattr(vector_store, "get_redis", lambda: type("R", (), {"incr": lambda self, key: broken()})())

    assert store.write(_chunks()) == 2
    assert "Could not bump" in caplog.text


def test_metadata_mappings_cover_every_chunk_metadata_field_with_the_right_type():
    mappings = metadata_mappings()

    assert set(ChunkMetadata.model_fields) | {"modality", "page", "timestamp"} == set(mappings)
    assert mappings["page"]["type"] == "integer"
    assert mappings["chunk_index"]["type"] == "integer"
    assert mappings["duration_s"]["type"] == "float"
    assert mappings["created"]["type"] == mappings["ingested_at"]["type"] == "date"
    assert mappings["doc_id"]["type"] == mappings["content"]["type"] == mappings["file_type"]["type"] == "keyword"
    assert mappings["title"]["type"] == "text" and "keyword" in mappings["title"]["fields"]


def test_document_metadata_adds_modality_page_timestamp_and_skips_missing():
    chunk = Chunk(text="t", source="/v.mp4", modality=Modality.VIDEO_FRAME, timestamp=5.0, metadata={"doc_id": "x"})

    assert document_metadata(chunk) == {"doc_id": "x", "modality": "video_frame", "timestamp": 5.0}


def test_real_stores_use_langchain_with_the_right_fields(monkeypatch):
    from elasticsearch import Elasticsearch

    # A real client object: building the LangChain stores sends no request, so no Elasticsearch is needed.
    monkeypatch.setattr(vector_store, "get_es_client", lambda: Elasticsearch("http://localhost:9"))

    store = VectorStore(Settings())

    assert store.text_store.vector_query_field == "text_embedding"
    assert store.image_store.vector_query_field == "image_embedding"
    assert store.text_store.query_field == store.image_store.query_field == "text"
