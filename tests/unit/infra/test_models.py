import pytest

from search_engine.core.config import Settings
from search_engine.infra import models


class _FakeSentenceTransformer:
    def __init__(self, name, device):
        self.name, self.device = name, device

    def get_embedding_dimension(self):
        return 384


@pytest.fixture(autouse=True)
def fake_sentence_transformers(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(models, "_device", lambda device: "cpu")
    models.get_text_model.cache_clear()
    yield
    models.get_text_model.cache_clear()


def test_text_model_loads_once_and_is_reused(monkeypatch):
    monkeypatch.setattr(models, "get_settings", lambda: Settings(text_embedding_model="bge", text_embedding_dim=384))

    first, second = models.get_text_model(), models.get_text_model()

    assert first is second
    assert (first.name, first.device) == ("bge", "cpu")


def test_dimension_mismatch_with_config_fails_clearly(monkeypatch):
    monkeypatch.setattr(models, "get_settings", lambda: Settings(text_embedding_dim=768))

    with pytest.raises(ValueError, match="384-dim vectors but text_embedding_dim is 768"):
        models.get_text_model()
