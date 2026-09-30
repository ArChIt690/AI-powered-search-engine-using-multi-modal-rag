"""Part 3 "Done when": every answer passes through LLM (with MCP / TOOLS) -> GUARDRAIL -> EVAL -> RESULT; unsafe
answers come back with a reason; safe results are cached; and every eval result is stored separately (and does not
appear in search results).

Real bge, CLIP and cross encoder, real Elasticsearch and Redis, a real MCP server (stdio), the real agent loop,
tools, guardrail rules, eval store and caches. Only the LLMs' words are scripted, so the test is repeatable and
spends no free-API quota. Run with `uv run pytest -m "slow and integration"`.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from search_engine.core.config import Settings
from search_engine.infra import models
from search_engine.infra.redis import get_redis
from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.llm.agent import LLMAgent
from search_engine.llm.eval import Evaluator
from search_engine.llm.eval_store import EvalStore
from search_engine.llm.guardrail import Guardrail
from search_engine.llm.mcp import load_mcp_tools
from search_engine.llm.pipeline import LLMPipeline
from search_engine.retrieval.hybrid_search import HybridSearch
from search_engine.retrieval.pipeline import SearchPipeline
from search_engine.retrieval.prompt_cache import PromptCache
from search_engine.retrieval.query_enhance import QueryEnhancer
from search_engine.retrieval.rerank import Reranker
from search_engine.schemas.eval import EvalScores, GuardrailVerdict
from search_engine.schemas.query import SearchRequest
from search_engine.schemas.response import AnswerSource
from tests.e2e.test_ingestion_done_when import _es, _SpeechStandIn, corpus  # noqa: F401  (corpus is a fixture)
from tests.e2e.test_retrieval_done_when import _TypedQueryLLM
from tests.unit.llm.fakes import ScriptedModel, tool_call
from tests.unit.llm.test_mcp import SERVER

pytestmark = [pytest.mark.slow, pytest.mark.integration]

MARKER = "zebrafinch"  # a word only the LLM's answers contain, to prove they never become search results


@pytest.fixture
def setup(corpus, monkeypatch):
    if _es() is None:
        pytest.skip("Elasticsearch is not running (docker compose up -d)")
    monkeypatch.setattr(models, "get_whisper", lambda: _SpeechStandIn())
    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        es_text_index=f"e2e_text_{suffix}", es_image_index=f"e2e_images_{suffix}",
        index_version_key=f"e2e:version:{suffix}", semantic_cache_dir=str(corpus / "faiss_cache"),
        eval_results_dir=str(corpus / "eval_results"),  # inside the ingested folder, on purpose
        mcp_servers={"glossary": SERVER},
    )
    ingestion = IngestionPipeline(settings)
    ingestion.ingest_path(corpus)

    llm_model = ScriptedModel(replies=[])
    reranker = Reranker(settings, search=HybridSearch(settings, text_embedder=ingestion.text_embedder))
    llm = LLMPipeline(
        settings,
        agent=LLMAgent(settings, models=[llm_model], reranker=reranker, mcp_tools=load_mcp_tools(settings)),
        guardrail=Guardrail(model=RunnableLambda(lambda prompt: GuardrailVerdict(safe=True))),
        evaluator=Evaluator(
            settings, model=RunnableLambda(lambda p: EvalScores(faithfulness=5, relevance=5, citation_correctness=5))
        ),
        store=EvalStore(settings),
    )
    prompt_cache = PromptCache(settings, embedder=ingestion.text_embedder, namespace=f"e2e{suffix}")
    pipeline = SearchPipeline(
        settings, enhancer=QueryEnhancer(settings, llm=_TypedQueryLLM()), reranker=reranker,
        prompt_cache=prompt_cache, llm=llm,
    )
    yield pipeline, llm_model, ingestion, corpus, settings
    _es().indices.delete(index=f"{settings.es_text_index},{settings.es_image_index}", ignore_unavailable=True)
    redis = get_redis()
    for key in [settings.index_version_key, *redis.scan_iter(f"e2e{suffix}:*")]:
        redis.delete(key)
    prompt_cache.similar.delete()


def test_answers_go_through_llm_tools_mcp_guardrail_eval_and_only_safe_results_are_cached(setup):
    pipeline, llm_model, ingestion, corpus, settings = setup
    store = pipeline.llm.store

    # 1. LLM with TOOLS and MCP -> GUARDRAIL (safe) -> EVAL (passed) -> RESULT, cached and stored
    llm_model.replies += [
        tool_call("search_documents", "c1", query="goroutines Go runtime"),
        tool_call("glossary_define", "c2", term="rrf"),
        tool_call("calculator", "c3", expression="12 * 2"),
        AIMessage(f"Revenue grew by twelve percent [1] ({MARKER})."),
    ]
    question = "How much did revenue grow in the quarterly review?"

    # top_k=3: the corpus is tiny, so a full context would leave the search tool nothing new to find
    first = pipeline.search(SearchRequest(query=question, top_k=3))

    assert first.source == AnswerSource.LLM and first.guardrail_reason is None
    assert first.tools_used == ["search_documents", "glossary_define", "calculator"]  # a local tool, MCP, a local tool
    assert first.eval.passed and first.citations[0].number == 1
    tool_results = [str(m.content) for m in llm_model.seen[-1] if m.type == "tool"]
    assert tool_results[0].startswith("[4] (notes.txt") and "Goroutines" in tool_results[0]  # a new, numbered passage
    assert "Reciprocal Rank Fusion" in tool_results[1] and tool_results[2] == "24"  # the MCP server, the calculator

    again = pipeline.search(SearchRequest(query=question, top_k=3))
    assert again.source == AnswerSource.FAISS_SEMANTIC_CACHE and again.answer == first.answer
    assert len(llm_model.seen) == 4  # the cached answer needed no LLM call

    # 2. GUARDRAIL: an UNSAFE answer comes back as the reason, and is not cached
    leaked = f"The deploy key is gsk_{'a1B2' * 8} [1] ({MARKER})."
    llm_model.replies += [AIMessage(leaked), AIMessage(leaked)]
    secret_question = "What is the deploy key for the Go runtime service?"

    blocked = pipeline.search(SearchRequest(query=secret_question))

    assert blocked.source == AnswerSource.GUARDRAIL_BLOCKED
    assert "API key" in blocked.guardrail_reason and "gsk_" not in blocked.answer
    assert blocked.citations == [] and blocked.eval is None
    assert pipeline.search(SearchRequest(query=secret_question)).source == AnswerSource.GUARDRAIL_BLOCKED  # not cached

    # 3. "all the eval results are stored separately": one record per LLM answer, with the verdicts
    records = store.read()
    assert [r.guardrail.safe for r in records] == [True, False, False]
    assert records[0].eval.passed and records[0].tools_used == first.tools_used and records[1].eval is None
    assert all(MARKER in r.answer for r in records)

    # ... and never in the Vector Database, even when the folder that holds them is ingested
    report = ingestion.ingest_path(corpus)
    assert any("eval_results" in path for path in report.skipped)
    for index in (settings.es_text_index, settings.es_image_index):
        assert _es().count(index=index, query={"match": {"text": MARKER}})["count"] == 0
