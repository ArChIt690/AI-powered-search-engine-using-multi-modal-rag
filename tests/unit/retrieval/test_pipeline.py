"""The Retrieval pipeline's order and decisions, with every box replaced by a fake that records its calls."""

import pytest
from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.retrieval import pipeline as pipeline_module
from search_engine.retrieval.pipeline import NO_RESULTS_ANSWER, SearchPipeline
from search_engine.schemas.query import EnhancedQuery, SearchFilters, SearchRequest
from search_engine.schemas.response import AnswerSource, CachedAnswer

CHUNKS = [
    Document("Goroutines are light threads.", metadata={"chunk_id": "d-0", "file_name": "go.md", "modality": "text",
                                                          "content": "text", "source": "/go.md"}),
    Document("chart from go.pdf, page 2", metadata={"chunk_id": "d-9", "file_name": "go.pdf", "modality": "image",
                                                     "content": "chart", "page": 2, "source": "/go.pdf"}),
]


class Recorder:
    def __init__(self, calls):
        self.calls = calls


class FakeEnhancer(Recorder):
    def enhance(self, query, history):
        self.calls.append(("enhance", query, len(history)))
        return EnhancedQuery(query=f"{query} (rewritten)", keywords=["go"], filters=SearchFilters(content=["text"]))


class FakeReranker(Recorder):
    def __init__(self, calls, chunks):
        super().__init__(calls)
        self.chunks = chunks

    def retrieve(self, query, filters, top_k):
        self.calls.append(("retrieve", query, filters, top_k))
        return self.chunks


class FakeCache(Recorder):
    def __init__(self, calls, name, hit=None, fail=False):
        super().__init__(calls)
        self.name, self.hit, self.fail, self.stored = name, hit, fail, []

    def lookup(self, question, key, version):
        self.calls.append((f"{self.name}.lookup", question, version))
        return self.hit

    def store(self, question, key, version, answer):
        self.stored.append((question, key, version, answer))


class FakeSessions(Recorder):
    def history(self, session_id):
        self.calls.append(("history", session_id))
        return ["earlier q", "earlier a"] if session_id else []

    def append(self, session_id, question, answer):
        self.calls.append(("append", session_id, question, answer))


class FakeAgent(Recorder):
    def __init__(self, calls, answer="Goroutines are light [1], see the chart [2].", error=None):
        super().__init__(calls)
        self.text, self.error = answer, error

    def answer(self, question, chunks, history):
        self.calls.append(("llm", question, len(chunks), len(history)))
        if self.error:
            raise self.error
        return self.text


def build(monkeypatch, *, chunks=CHUNKS, faiss_hit=None, redis_hit=None, agent_answer=None, agent_error=None,
          version=7):
    calls = []
    faiss, redis = FakeCache(calls, "faiss", faiss_hit), FakeCache(calls, "redis", redis_hit)
    agent = FakeAgent(calls, **({"answer": agent_answer} if agent_answer else {}), error=agent_error)
    if isinstance(version, Exception):
        monkeypatch.setattr(pipeline_module, "index_version", lambda settings: (_ for _ in ()).throw(version))
    else:
        monkeypatch.setattr(pipeline_module, "index_version", lambda settings: version)
    pipeline = SearchPipeline(
        Settings(), enhancer=FakeEnhancer(calls), reranker=FakeReranker(calls, chunks), semantic_cache=faiss,
        prompt_cache=redis, sessions=FakeSessions(calls), agent=agent,
    )
    return pipeline, calls, faiss, redis


def names(calls):
    return [call[0] for call in calls]


def test_a_miss_runs_every_box_in_the_diagrams_order_and_caches_the_answer(monkeypatch):
    pipeline, calls, faiss, redis = build(monkeypatch)

    response = pipeline.search(SearchRequest(query="goroutines?", session_id="s1", top_k=5,
                                             filters=SearchFilters(file_type=["md"])))

    assert names(calls) == ["history", "enhance", "retrieve", "faiss.lookup", "redis.lookup", "llm", "append"]
    assert calls[2][1] == "goroutines? (rewritten) go"  # hybrid search runs on the enhanced query + keywords
    assert calls[2][2] == SearchFilters(file_type=["md"], content=["text"])  # user filter + the query's filter
    assert calls[2][3] == 5
    assert response.source == AnswerSource.LLM and response.enhanced_query == "goroutines? (rewritten)"
    assert [(c.number, c.chunk_id, c.modality, c.page) for c in response.citations] == [
        (1, "d-0", "text", None), (2, "d-9", "image", 2)
    ]
    assert redis.stored[0][:3] == ("goroutines? (rewritten)", ["d-0", "d-9"], 7)
    assert faiss.stored[0][:3] == ("goroutines? (rewritten)", SearchFilters(file_type=["md"], content=["text"]), 7)
    assert calls[-1] == ("append", "s1", "goroutines?", response.answer)


def test_a_faiss_hit_skips_redis_and_the_llm(monkeypatch):
    cached = CachedAnswer(answer="cached [1]")
    pipeline, calls, _, _ = build(monkeypatch, faiss_hit=cached)

    response = pipeline.search(SearchRequest(query="q"))

    assert "llm" not in names(calls) and "redis.lookup" not in names(calls)
    assert response.source == AnswerSource.FAISS_SEMANTIC_CACHE and response.answer == "cached [1]"


def test_a_redis_hit_skips_the_llm(monkeypatch):
    pipeline, calls, faiss, _ = build(monkeypatch, redis_hit=(CachedAnswer(answer="from redis"), "similar"))

    response = pipeline.search(SearchRequest(query="q"))

    assert names(calls)[-2:] == ["redis.lookup", "append"] and "llm" not in names(calls)
    assert response.source == AnswerSource.REDIS_PROMPT_CACHE and faiss.stored == []


def test_no_chunks_means_no_llm_call_and_nothing_cached(monkeypatch):
    pipeline, calls, faiss, redis = build(monkeypatch, chunks=[])

    response = pipeline.search(SearchRequest(query="q"))

    assert response.source == AnswerSource.NO_RESULTS and response.answer == NO_RESULTS_ANSWER
    assert not {"faiss.lookup", "llm"} & set(names(calls)) and faiss.stored == redis.stored == []


def test_an_answer_without_citations_is_not_cached(monkeypatch):
    pipeline, _, faiss, redis = build(monkeypatch, agent_answer="The documents don't cover this.")

    response = pipeline.search(SearchRequest(query="q"))

    assert response.citations == [] and faiss.stored == redis.stored == []


def test_without_the_index_version_the_caches_are_skipped(monkeypatch):
    pipeline, calls, faiss, _ = build(monkeypatch, version=ConnectionError("redis down"))

    response = pipeline.search(SearchRequest(query="q"))

    assert response.source == AnswerSource.LLM
    assert "faiss.lookup" not in names(calls) and faiss.stored == []


def test_an_llm_failure_is_raised_and_not_saved_to_the_session(monkeypatch):
    pipeline, calls, faiss, _ = build(monkeypatch, agent_error=LLMUnavailableError("429"))

    with pytest.raises(LLMUnavailableError):
        pipeline.search(SearchRequest(query="q", session_id="s1"))

    assert "append" not in names(calls) and faiss.stored == []
