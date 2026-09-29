import pytest

from search_engine.core.config import Settings
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.retrieval.semantic_cache import SemanticCache
from search_engine.schemas.query import SearchFilters
from search_engine.schemas.response import CachedAnswer, Citation

QUESTION = "What are the recommended dimensions for a horizontal HD video ad?"
ANSWER = CachedAnswer(
    answer="1920 x 1080 [1].",
    citations=[Citation(number=1, chunk_id="d-1", file_name="specs.pdf", source="/specs.pdf", modality="text",
                        content="text", page=7, snippet="Horizontal: 1920 x 1080")],
)


@pytest.fixture
def cache(tmp_path, fake_models):
    """With the fake bge, the same text gives the same vector and different texts are unrelated."""
    settings = Settings(semantic_cache_dir=str(tmp_path / "faiss"), semantic_cache_max_entries=3)
    return SemanticCache(settings, embedder=TextEmbedder(settings))


def test_empty_cache_is_a_miss(cache):
    assert cache.lookup(QUESTION, SearchFilters(), index_version=1) is None


def test_the_same_question_is_a_hit_with_its_citations(cache):
    cache.store(QUESTION, SearchFilters(), 1, ANSWER)

    assert cache.lookup(QUESTION, SearchFilters(), 1) == ANSWER


def test_a_different_question_is_a_miss(cache):
    cache.store(QUESTION, SearchFilters(), 1, ANSWER)

    assert cache.lookup("How do goroutines work?", SearchFilters(), 1) is None


def test_other_filters_are_a_miss_and_filter_order_does_not_matter(cache):
    cache.store(QUESTION, SearchFilters(file_type=["pdf", "md"]), 1, ANSWER)

    assert cache.lookup(QUESTION, SearchFilters(), 1) is None
    assert cache.lookup(QUESTION, SearchFilters(file_type=["mp4"]), 1) is None
    assert cache.lookup(QUESTION, SearchFilters(file_type=["md", "pdf"]), 1) == ANSWER


def test_a_new_ingest_empties_the_cache(cache):
    cache.store(QUESTION, SearchFilters(), 1, ANSWER)

    assert cache.lookup(QUESTION, SearchFilters(), index_version=2) is None
    assert cache.lookup(QUESTION, SearchFilters(), index_version=2) is None  # and it stays empty


def test_the_cache_survives_a_restart(cache):
    cache.store(QUESTION, SearchFilters(), 5, ANSWER)

    reopened = SemanticCache(cache.settings, embedder=cache.embedder)

    assert reopened.lookup(QUESTION, SearchFilters(), 5) == ANSWER
    assert reopened.lookup(QUESTION, SearchFilters(), 6) is None  # the saved cache is for an older corpus


def test_a_full_cache_is_cleared_before_the_next_write(cache):
    for i in range(3):
        cache.store(f"question {i}", SearchFilters(), 1, ANSWER)

    cache.store(QUESTION, SearchFilters(), 1, ANSWER)

    assert cache.lookup("question 0", SearchFilters(), 1) is None
    assert cache.lookup(QUESTION, SearchFilters(), 1) == ANSWER


def test_a_failing_cache_is_a_miss_not_an_error(cache, monkeypatch):
    cache.store(QUESTION, SearchFilters(), 1, ANSWER)
    monkeypatch.setattr(cache.embedder, "embed_query", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cache.lookup(QUESTION, SearchFilters(), 1) is None
    cache.store("another", SearchFilters(), 1, ANSWER)  # doesn't raise either
