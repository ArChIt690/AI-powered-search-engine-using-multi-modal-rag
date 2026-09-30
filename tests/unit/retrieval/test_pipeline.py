"""The Retrieval pipeline's order and decisions, with every box replaced by a fake that records its calls."""

import pytest
from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.retrieval import pipeline as pipeline_module
from search_engine.retrieval.pipeline import NO_RESULTS_ANSWER, SearchPipeline
from search_engine.llm.pipeline import LLMResult
from search_engine.schemas.eval import EvalResult, GuardrailVerdict
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


class FakeLLM(Recorder):
    """Stands in for the LLM Architecture (llm/pipeline.py): returns a RESULT with the given verdicts."""

    def __init__(self, calls, answer="Goroutines are light [1], see the chart [2].", error=None,
                 guardrail=GuardrailVerdict(safe=True), eval_passed=True, extra_chunks=()):
        super().__init__(calls)
        self.text, self.error, self.guardrail, self.eval_passed = answer, error, guardrail, eval_passed
        self.extra_chunks = list(extra_chunks)

    def run(self, question, chunks, history, filters):
        self.calls.append(("llm", question, len(chunks), len(history)))
        if self.error:
            raise self.error
        scores = None
        if self.guardrail.safe and self.eval_passed is not None:
            scores = EvalResult(faithfulness=5, relevance=5, citation_correctness=5, passed=self.eval_passed)
        return LLMResult(self.text, [*chunks, *self.extra_chunks], self.guardrail, scores, ["calculator"])


def build(monkeypatch, *, chunks=CHUNKS, faiss_hit=None, redis_hit=None, version=7, **llm):
    calls = []
    faiss, redis = FakeCache(calls, "faiss", faiss_hit), FakeCache(calls, "redis", redis_hit)
    if isinstance(version, Exception):
        monkeypatch.setattr(pipeline_module, "index_version", lambda settings: (_ for _ in ()).throw(version))
    else:
        monkeypatch.setattr(pipeline_module, "index_version", lambda settings: version)
    pipeline = SearchPipeline(
        Settings(), enhancer=FakeEnhancer(calls), reranker=FakeReranker(calls, chunks), semantic_cache=faiss,
        prompt_cache=redis, sessions=FakeSessions(calls), llm=FakeLLM(calls, **llm),
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
    assert response.eval.passed and response.tools_used == ["calculator"] and response.guardrail_reason is None


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
    pipeline, _, faiss, redis = build(monkeypatch, answer="The documents don't cover this.")

    response = pipeline.search(SearchRequest(query="q"))

    assert response.citations == [] and faiss.stored == redis.stored == []


def test_without_the_index_version_the_caches_are_skipped(monkeypatch):
    pipeline, calls, faiss, _ = build(monkeypatch, version=ConnectionError("redis down"))

    response = pipeline.search(SearchRequest(query="q"))

    assert response.source == AnswerSource.LLM
    assert "faiss.lookup" not in names(calls) and faiss.stored == []


def test_an_llm_failure_is_raised_and_not_saved_to_the_session(monkeypatch):
    pipeline, calls, faiss, _ = build(monkeypatch, error=LLMUnavailableError("429"))

    with pytest.raises(LLMUnavailableError):
        pipeline.search(SearchRequest(query="q", session_id="s1"))

    assert "append" not in names(calls) and faiss.stored == []


def test_an_unsafe_answer_is_replaced_by_the_reason_and_never_cached(monkeypatch):
    unsafe = GuardrailVerdict(safe=False, category="secrets", reason="The answer contained an API key.")
    pipeline, calls, faiss, redis = build(monkeypatch, guardrail=unsafe, answer="The key is gsk_secret [1].")

    response = pipeline.search(SearchRequest(query="what is the key?", session_id="s1"))

    assert response.source == AnswerSource.GUARDRAIL_BLOCKED
    assert response.guardrail_reason == "The answer contained an API key."
    assert "gsk_secret" not in response.answer and "The answer contained an API key." in response.answer
    assert response.citations == [] and response.eval is None
    assert faiss.stored == redis.stored == []
    assert "gsk_secret" not in calls[-1][3]  # the session keeps the refusal, not the unsafe text


@pytest.mark.parametrize("eval_passed", [False, None])  # failed EVAL, or the judge could not run
def test_an_answer_that_did_not_pass_eval_is_returned_but_not_cached(monkeypatch, eval_passed):
    pipeline, _, faiss, redis = build(monkeypatch, eval_passed=eval_passed)

    response = pipeline.search(SearchRequest(query="q"))

    assert response.source == AnswerSource.LLM and len(response.citations) == 2
    assert faiss.stored == redis.stored == []


def test_passages_a_tool_added_can_be_cited(monkeypatch):
    found = Document("South had 95.", metadata={"chunk_id": "d-7", "file_name": "go.md", "modality": "text",
                                                 "content": "text", "source": "/go.md"})
    pipeline, _, _, _ = build(monkeypatch, answer="South had 95 [3].", extra_chunks=[found])

    response = pipeline.search(SearchRequest(query="q"))

    assert [(c.number, c.chunk_id) for c in response.citations] == [(3, "d-7")]
