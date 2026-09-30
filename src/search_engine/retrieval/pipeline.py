"""The Retrieval pipeline, in the diagram's order, for one question (`/search`, `search-engine search`).

USER -> QUERY -> QUERY Enhancement -> METADATA FILTERING -> ELASTICSEARCH HYBRID SEARCH -> RERANKING
     -> FAISS SEMANTIC CACHE (HIT -> USER) -> REDIS PROMPT CACHING (HIT -> USER)
     -> LLM Architecture (llm/pipeline.py: LLM -> GUARDRAIL -> EVAL -> RESULT) -> caches + USER
USER -- Sessional Queries --> REDIS
"""

import logging

from langchain_core.documents import Document

from search_engine.core.config import Settings, get_settings
from search_engine.infra.redis import index_version
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.llm.agent import cited_numbers
from search_engine.llm.pipeline import LLMPipeline
from search_engine.retrieval.filters import merge_filters
from search_engine.retrieval.hybrid_search import HybridSearch
from search_engine.retrieval.prompt_cache import PromptCache
from search_engine.retrieval.query_enhance import QueryEnhancer
from search_engine.retrieval.rerank import Reranker
from search_engine.retrieval.semantic_cache import SemanticCache
from search_engine.retrieval.session import SessionStore
from search_engine.schemas.query import SearchRequest
from search_engine.schemas.eval import EvalResult
from search_engine.schemas.response import AnswerSource, CachedAnswer, Citation, SearchResponse

logger = logging.getLogger(__name__)

NO_RESULTS_ANSWER = "No documents matched your question and filters, so there is nothing to answer from."
BLOCKED_ANSWER = "The answer was blocked by the safety check: {reason}"
_SNIPPET_CHARS = 300


class SearchPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        enhancer: QueryEnhancer | None = None,
        reranker: Reranker | None = None,
        semantic_cache: SemanticCache | None = None,
        prompt_cache: PromptCache | None = None,
        sessions: SessionStore | None = None,
        llm: LLMPipeline | None = None,
    ):
        self.settings = s = settings or get_settings()
        embedder = TextEmbedder(s)  # one bge for query embedding and both caches
        self.enhancer = enhancer or QueryEnhancer(s)
        self.reranker = reranker or Reranker(s, search=HybridSearch(s, text_embedder=embedder))
        self.semantic_cache = semantic_cache or SemanticCache(s, embedder=embedder)
        self.prompt_cache = prompt_cache or PromptCache(s, embedder=embedder)
        self.sessions = sessions or SessionStore(s)
        self.llm = llm or LLMPipeline(s, reranker=self.reranker)  # its search tool reuses this reranker

    def search(self, request: SearchRequest) -> SearchResponse:
        """One question in, the FINAL RESULT out. Raises LLMUnavailableError when the LLM is needed and fails."""
        history = self.sessions.history(request.session_id)
        enhanced = self.enhancer.enhance(request.query, history)
        filters = merge_filters(request.filters, enhanced.filters)
        chunks = self.reranker.retrieve(enhanced.search_text, filters, request.top_k)

        def respond(
            result: CachedAnswer,
            source: AnswerSource,
            *,
            guardrail_reason: str | None = None,
            eval: EvalResult | None = None,
            tools_used: list[str] | None = None,
        ) -> SearchResponse:
            self.sessions.append(request.session_id, request.query, result.answer)
            return SearchResponse(
                query=request.query, enhanced_query=enhanced.query, answer=result.answer,
                citations=result.citations, source=source, session_id=request.session_id,
                guardrail_reason=guardrail_reason, eval=eval, tools_used=tools_used or [],
            )

        if not chunks:
            return respond(CachedAnswer(answer=NO_RESULTS_ANSWER), AnswerSource.NO_RESULTS)

        version = self._index_version()
        chunk_ids = [doc.metadata["chunk_id"] for doc in chunks]
        if version is not None:
            if cached := self.semantic_cache.lookup(enhanced.query, filters, version):
                return respond(cached, AnswerSource.FAISS_SEMANTIC_CACHE)
            if hit := self.prompt_cache.lookup(enhanced.query, chunk_ids, version):
                return respond(hit[0], AnswerSource.REDIS_PROMPT_CACHE)

        result = self.llm.run(enhanced.query, chunks, history, filters)
        if not result.safe:  # GUARDRAIL: UNSAFE -> USER, "sends the reason alongside"; nothing is cached
            blocked = CachedAnswer(answer=BLOCKED_ANSWER.format(reason=result.guardrail.reason))
            return respond(blocked, AnswerSource.GUARDRAIL_BLOCKED, guardrail_reason=result.guardrail.reason)
        answer = CachedAnswer(answer=result.answer, citations=citations(result.answer, result.chunks))
        # RESULT "goes for caching": only when it passed GUARDRAIL and EVAL, and cites something worth reusing
        if version is not None and result.cacheable and answer.citations:
            self.prompt_cache.store(enhanced.query, chunk_ids, version, answer)
            self.semantic_cache.store(enhanced.query, filters, version, answer)
        return respond(answer, AnswerSource.LLM, eval=result.eval, tools_used=result.tools_used)

    def _index_version(self) -> int | None:
        """None when Redis is unreachable: then the caches are skipped, as a stale answer can't be ruled out."""
        try:
            return index_version(self.settings)
        except Exception as error:
            logger.warning("Could not read the index version, skipping the caches: %s", error)
            return None


def citations(answer: str, chunks: list[Document]) -> list[Citation]:
    """The chunks the answer cites as [n], with where each one is (file, page, video time)."""
    result = []
    for number in cited_numbers(answer, len(chunks)):
        m = chunks[number - 1].metadata
        result.append(
            Citation(
                number=number, chunk_id=m["chunk_id"], file_name=m.get("file_name", ""), source=m.get("source", ""),
                modality=m.get("modality", "text"), content=m.get("content", "text"), page=m.get("page"),
                timestamp=m.get("timestamp"), snippet=chunks[number - 1].page_content[:_SNIPPET_CHARS],
            )
        )
    return result
