import numpy as np
import pytest

from search_engine.core.config import Settings
from search_engine.infra import models
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.schemas.chunk import Chunk

DIM = 384


class _FakeModel:
    """Stands in for bge: vector[0] = text length, so each text gets its own recognisable vector."""

    def __init__(self):
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        assert normalize_embeddings and convert_to_numpy
        self.calls.append((list(texts), batch_size))
        vectors = np.zeros((len(texts), DIM), dtype=np.float32)
        vectors[:, 0] = [len(t) for t in texts]
        return vectors


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(models, "get_text_model", lambda: model)
    return model


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, source="/docs/a.txt")


def test_embed_chunks_fills_vectors_in_order(fake_model):
    chunks = [_chunk("a"), _chunk("bbb"), _chunk("cc")]

    result = TextEmbedder(Settings(embedding_batch_size=16)).embed_chunks(chunks)

    assert result is chunks
    assert [c.text_embedding[0] for c in chunks] == [1.0, 3.0, 2.0]
    assert all(len(c.text_embedding) == DIM for c in chunks)
    assert fake_model.calls == [(["a", "bbb", "cc"], 16)]  # one call; the model batches internally


def test_empty_input_does_not_load_the_model(monkeypatch):
    monkeypatch.setattr(models, "get_text_model", lambda: pytest.fail("model should not load"))

    assert TextEmbedder().embed_chunks([]) == []
    assert TextEmbedder().embed([]).shape == (0, DIM)


def test_query_gets_bge_prefix_but_documents_do_not(fake_model):
    embedder = TextEmbedder(Settings(text_query_instruction="Q: "))

    embedder.embed_documents(["a doc"])
    embedder.embed_query("a question")

    assert [texts for texts, _ in fake_model.calls] == [["a doc"], ["Q: a question"]]
