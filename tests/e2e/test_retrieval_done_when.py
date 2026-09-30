"""Part 2 "Done when": a question returns a cited answer produced from reranked hybrid results (text and image
hits), and asking the same or a similar question again is answered from the cache without calling the LLM.

Real bge, CLIP and cross encoder, real Elasticsearch and Redis (docker compose up -d). The two LLM calls are
stand-ins, so the test is repeatable and never spends free-API quota: Query Enhancement returns the query as
typed, and the answer LLM counts its calls and cites every passage it gets. Run with
`uv run pytest -m "slow and integration"`.
"""

import uuid

import pytest
from langchain_core.runnables import RunnableLambda

from search_engine.core.config import Settings
from search_engine.infra import models
from search_engine.infra.redis import get_redis
from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.llm.pipeline import LLMResult
from search_engine.retrieval.pipeline import SearchPipeline
from search_engine.retrieval.prompt_cache import PromptCache
from search_engine.retrieval.query_enhance import QueryEnhancer
from search_engine.schemas.eval import EvalResult, GuardrailVerdict
from search_engine.schemas.query import EnhancedQuery, SearchRequest
from search_engine.schemas.response import AnswerSource
from tests.e2e.test_ingestion_done_when import _es, _SpeechStandIn, corpus  # noqa: F401  (corpus is a fixture)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _TypedQueryLLM:
    """Query Enhancement stand-in: the query as typed, no keywords or filters."""

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda prompt: EnhancedQuery(query=prompt.to_string().rsplit("Latest question: ", 1)[1]))


class _CountingLLM:
    """Stands in for the LLM Architecture: counts its calls and cites every passage it gets (safe, eval passed)."""

    def __init__(self):
        self.calls = 0

    def run(self, question, chunks, history=(), filters=None):
        self.calls += 1
        answer = "From the documents: " + " ".join(f"[{n}]" for n in range(1, len(chunks) + 1))
        scores = EvalResult(faithfulness=5, relevance=5, citation_correctness=5, passed=True)
        return LLMResult(answer, list(chunks), GuardrailVerdict(safe=True), scores)


@pytest.fixture
def setup(corpus, tmp_path, monkeypatch):
    if _es() is None:
        pytest.skip("Elasticsearch is not running (docker compose up -d)")
    monkeypatch.setattr(models, "get_whisper", lambda: _SpeechStandIn())
    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        es_text_index=f"e2e_text_{suffix}", es_image_index=f"e2e_images_{suffix}",
        index_version_key=f"e2e:version:{suffix}", semantic_cache_dir=str(tmp_path / "faiss"),
    )
    ingestion = IngestionPipeline(settings)
    ingestion.ingest_path(corpus)
    agent = _CountingLLM()
    prompt_cache = PromptCache(settings, embedder=ingestion.text_embedder, namespace=f"e2e{suffix}")
    pipeline = SearchPipeline(
        settings, enhancer=QueryEnhancer(settings, llm=_TypedQueryLLM()), prompt_cache=prompt_cache, llm=agent
    )
    yield pipeline, agent, ingestion, corpus
    _es().indices.delete(index=f"{settings.es_text_index},{settings.es_image_index}", ignore_unavailable=True)
    redis = get_redis()
    for key in [settings.index_version_key, *redis.scan_iter(f"e2e{suffix}:*")]:
        redis.delete(key)
    prompt_cache.similar.delete()


def test_cited_answer_from_text_and_image_hits_then_served_from_the_cache(setup):
    pipeline, agent, ingestion, corpus = setup
    question = "How much did revenue grow in the quarterly review?"

    first = pipeline.search(SearchRequest(query=question))

    assert first.source == AnswerSource.LLM and agent.calls == 1
    modalities = {citation.modality.value for citation in first.citations}
    assert "video_transcript" in modalities  # the spoken "Revenue grew by twelve percent" ranks among the text hits
    assert modalities & {"image", "video_frame"}  # and picture hits are part of the reranked context
    assert all(c.file_name and c.chunk_id for c in first.citations)

    again = pipeline.search(SearchRequest(query=question))
    similar = pipeline.search(SearchRequest(query="How much did the revenue grow in the quarterly review?"))

    assert again.source == similar.source == AnswerSource.FAISS_SEMANTIC_CACHE
    assert again.answer == first.answer and again.citations == first.citations
    assert agent.calls == 1  # neither called the LLM

    # a new ingest changes the corpus: cached answers are no longer trusted
    (corpus / "extra.txt").write_text("Revenue grew by twelve percent again next quarter.", encoding="utf-8")
    ingestion.ingest_path(corpus / "extra.txt")

    after_ingest = pipeline.search(SearchRequest(query=question))

    assert after_ingest.source == AnswerSource.LLM and agent.calls == 2
