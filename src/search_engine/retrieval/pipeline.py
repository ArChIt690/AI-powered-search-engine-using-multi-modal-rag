"""The Retrieval pipeline, in the diagram's order, for one question (`/search`, `search-engine search`).

USER -> QUERY -> QUERY Enhancement -> METADATA FILTERING -> ELASTICSEARCH HYBRID SEARCH -> RERANKING
     -> FAISS SEMANTIC CACHE (HIT -> USER) -> REDIS PROMPT CACHING (HIT -> USER) -> LLM -> USER
USER -- Sessional Queries --> REDIS
"""

import logging

from langchain_core.documents import Document

from search_engine.core.config import Settings, get_settings
from search_engine.infra.redis import index_version
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.llm.agent import LLMAgent, cited_numbers
from search_engine.retrieval.filters import merge_filters
from search_engine.retrieval.hybrid_search import HybridSearch
from search_engine.retrieval.prompt_cache import PromptCache
from search_engine.retrieval.query_enhance import QueryEnhancer
from search_engine.retrieval.rerank import Reranker
from search_engine.retrieval.semantic_cache import SemanticCache
from search_engine.retrieval.session import SessionStore
from search_engine.schemas.query import SearchRequest
from search_engine.schemas.response import AnswerSource, CachedAnswer, Citation, SearchResponse

logger = logging.getLogger(__name__)

NO_RESULTS_ANSWER = "No documents matched your question and filters, so there is nothing to answer from."
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
        agent: LLMAgent | None = None,
    ):
        self.settings = s = settings or get_settings()
        embedder = TextEmbedder(s)  # one bge for query embedding and both caches
        self.enhancer = enhancer or QueryEnhancer(s)
        self.reranker = reranker or Reranker(s, search=HybridSearch(s, text_embedder=embedder))
        self.semantic_cache = semantic_cache or SemanticCache(s, embedder=embedder)
        self.prompt_cache = prompt_cache or PromptCache(s, embedder=embedder)
        self.sessions = sessions or SessionStore(s)
        self.agent = agent or LLMAgent()

    def search(self, request: SearchRequest) -> SearchResponse:
        """One question in, a cited answer out. Raises LLMUnavailableError when the LLM is needed and fails."""
        history = self.sessions.history(request.session_id)
        enhanced = self.enhancer.enhance(request.query, history)
        filters = merge_filters(request.filters, enhanced.filters)
        chunks = self.reranker.retrieve(enhanced.search_text, filters, request.top_k)

        def respond(result: CachedAnswer, source: AnswerSource) -> SearchResponse:
            self.sessions.append(request.session_id, request.query, result.answer)
            return SearchResponse(
                query=request.query, enhanced_query=enhanced.query, answer=result.answer,
                citations=result.citations, source=source, session_id=request.session_id,
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

        text = self.agent.answer(enhanced.query, chunks, history)
        result = CachedAnswer(answer=text, citations=citations(text, chunks))
        if version is not None and result.citations:  # an answer that cites nothing isn't worth reusing
            self.prompt_cache.store(enhanced.query, chunk_ids, version, result)
            self.semantic_cache.store(enhanced.query, filters, version, result)  # Part 3: after GUARDRAIL + EVAL
        return respond(result, AnswerSource.LLM)

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
