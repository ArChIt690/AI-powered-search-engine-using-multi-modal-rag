"""REDIS PROMPT CACHING and Sessional Queries against real Redis Stack (docker compose up -d); skipped when down.

A stand-in query embedder maps words to fixed axes, so "similar question" is under the test's control.
"""

import uuid

import pytest
from langchain_core.embeddings import Embeddings

from search_engine.core.config import Settings
from search_engine.infra.redis import get_redis
from search_engine.retrieval.prompt_cache import PromptCache
from search_engine.retrieval.session import SessionStore
from search_engine.schemas.response import CachedAnswer

pytestmark = pytest.mark.integration


def _redis_up() -> bool:
    try:
        return get_redis().ping()
    except Exception:
        return False


if not _redis_up():
    pytest.skip("Redis is not running (docker compose up -d)", allow_module_level=True)


class _AxisEmbeddings(Embeddings):
    """Questions about revenue point one way, questions about goroutines another."""

    def embed_query(self, text):
        vector = [0.0] * 384
        vector[0 if "revenue" in text.lower() else 1] = 1.0
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


ANSWER = CachedAnswer(answer="North leads [1].")
QUESTION = "Which region had the highest revenue?"


@pytest.fixture
def cache():
    namespace = f"test{uuid.uuid4().hex[:8]}"
    cache = PromptCache(Settings(prompt_cache_ttl_s=600), embedder=_AxisEmbeddings(), namespace=namespace)
    yield cache
    redis = get_redis()
    for key in redis.scan_iter(f"{namespace}:prompt*"):
        redis.delete(key)
    cache.similar.delete()  # drops the test's search index and its entries


def test_empty_cache_is_a_miss(cache):
    assert cache.lookup(QUESTION, ["c1"], index_version=1) is None


def test_the_same_prompt_is_an_exact_hit(cache):
    cache.store(QUESTION, ["c1", "c2"], 1, ANSWER)

    assert cache.lookup(QUESTION, ["c1", "c2"], 1) == (ANSWER, "exact")


def test_a_similar_question_with_the_same_chunks_is_a_similar_hit(cache):
    cache.store(QUESTION, ["c1", "c2"], 1, ANSWER)

    assert cache.lookup("Top region by revenue?", ["c2", "c1"], 1) == (ANSWER, "similar")


def test_other_chunks_another_question_or_a_new_corpus_version_miss(cache):
    cache.store(QUESTION, ["c1", "c2"], 1, ANSWER)

    assert cache.lookup("Top region by revenue?", ["c3"], 1) is None  # other evidence
    assert cache.lookup("How do goroutines work?", ["c1", "c2"], 1) is None  # not similar
    assert cache.lookup(QUESTION, ["c1", "c2"], 2) is None  # a new ingest happened


def test_similar_lookup_finds_the_entry_with_the_same_chunks_even_when_a_closer_one_has_others(cache):
    cache.store(QUESTION, ["other"], 1, CachedAnswer(answer="from other chunks"))
    cache.store("Top region by revenue?", ["c1"], 1, ANSWER)

    assert cache.lookup(QUESTION + " please", ["c1"], 1) == (ANSWER, "similar")


def test_entries_expire(cache):
    cache.store(QUESTION, ["c1"], 1, ANSWER)

    ttls = [get_redis().ttl(key) for key in get_redis().scan_iter(f"{cache.namespace}*")]
    assert ttls and all(0 < ttl <= 600 for ttl in ttls)


def test_session_history_is_kept_trimmed_and_expires():
    sessions = SessionStore(Settings(session_history_turns=2, session_ttl_s=120))
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        for i in range(3):
            sessions.append(session_id, f"question {i}", f"answer {i}")

        history = sessions.history(session_id)

        assert [m.text for m in history] == ["question 1", "answer 1", "question 2", "answer 2"]
        assert [type(m).__name__ for m in history[:2]] == ["HumanMessage", "AIMessage"]
        assert 0 < get_redis().ttl(f"search:session:{session_id}") <= 120
        assert sessions.history(None) == [] and sessions.history("no-such-session") == []
    finally:
        get_redis().delete(f"search:session:{session_id}")
