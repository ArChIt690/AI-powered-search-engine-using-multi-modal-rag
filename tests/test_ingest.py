"""End-to-end ingestion against the real Elasticsearch from docker-compose (skipped if it isn't running).

Uses a throwaway index and a tiny fake embedder, so no model download is needed.
"""

import uuid

import numpy as np
import pytest
from elasticsearch import Elasticsearch

from Search_Engine.config import Settings
from Search_Engine.Data.ingest import IngestionPipeline

DIM = 4


class FakeEmbedder:
    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        vectors = rng.random((len(texts), DIM), dtype=np.float32) + 0.1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_chunks(self, chunks):
        for chunk, vector in zip(chunks, self.embed([c.text for c in chunks])):
            chunk.text_embedding = vector.tolist()
        return chunks


@pytest.fixture
def es():
    client = Elasticsearch("http://localhost:9200")
    if not client.ping():
        pytest.skip("Elasticsearch is not running (docker compose up -d)")
    return client


@pytest.fixture
def pipeline(es):
    index = f"test_chunks_{uuid.uuid4().hex[:8]}"
    settings = Settings(es_index=index, text_embedding_dim=DIM, chunk_size=200, chunk_overlap=20)
    yield IngestionPipeline(settings=settings, es=es, embedder=FakeEmbedder())
    es.indices.delete(index=index, ignore_unavailable=True)


def test_ingest_folder_end_to_end(pipeline, es, tmp_path):
    (tmp_path / "notes.md").write_text("# Cats\nCats sleep a lot.\n\n# Dogs\nDogs like walks.", encoding="utf-8")
    (tmp_path / "people.csv").write_text("name,city\nAlice,Paris\nBob,Rome\n", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"")

    report = pipeline.ingest_path(tmp_path)

    assert report.files == 2
    assert report.skipped == [str(tmp_path / "photo.png")]
    assert report.failed == {}

    index = pipeline.settings.es_index
    hits = es.search(index=index, query={"match": {"text": "walks"}})["hits"]["hits"]
    assert len(hits) == 1
    doc = hits[0]["_source"]
    assert doc["metadata"]["section"] == "Dogs"
    assert doc["metadata"]["file_type"] == "md"
    assert len(doc["text_embedding"]) == DIM


def test_reingesting_a_file_replaces_its_chunks(pipeline, es, tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name\n" + "\n".join(f"person {i} with a fairly long description" for i in range(20)), encoding="utf-8")
    pipeline.ingest_path(path)

    path.write_text("name\nonly one person now\n", encoding="utf-8")
    pipeline.ingest_path(path)

    index = pipeline.settings.es_index
    assert es.count(index=index)["count"] == 1
