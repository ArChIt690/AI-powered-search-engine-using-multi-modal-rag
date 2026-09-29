"""REDIS PROMPT CACHING: "checks if exact prompt or similar prompt there or not". HIT -> USER, MISS -> LLM.

The prompt is what the LLM would be sent: the question after QUERY Enhancement + the reranked chunks (by id).
- exact:   the same question with the same chunks (LangChain `RedisCache`, a hash lookup)
- similar: a similar question (bge, cosine distance <= prompt_cache_distance) with exactly the same chunks
           (redisvl `SemanticCache`, the library under langchain-redis, filtered by a tag of the chunk ids)
Unlike the FAISS cache, a HIT here needs the same evidence, so the similar match can be looser.
Entries carry the LLM model and the corpus version, so a new ingest or another model is a MISS; they expire by TTL.
"""

import hashlib
import logging

from langchain_core.outputs import Generation
from langchain_redis import RedisCache
from langchain_redis.cache import EmbeddingsVectorizer
from redisvl.extensions.cache.llm import SemanticCache as RedisVLSemanticCache
from redisvl.query.filter import Tag

from search_engine.core.config import Settings, get_settings
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.schemas.response import CachedAnswer

logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(
        self, settings: Settings | None = None, *, embedder: TextEmbedder | None = None, namespace: str = "search"
    ):
        self.settings = settings or get_settings()
        s = self.settings
        self.namespace = namespace  # Redis key prefix; tests use their own
        self.exact = RedisCache(redis_url=s.redis_url, ttl=s.prompt_cache_ttl_s, prefix=f"{namespace}:prompt")
        self._embedder = embedder or TextEmbedder(s)
        self._similar: RedisVLSemanticCache | None = None  # created on first use: it builds a Redis search index

    @property
    def similar(self) -> RedisVLSemanticCache:
        if self._similar is None:
            self._similar = RedisVLSemanticCache(
                name=f"{self.namespace}_prompt_similar",
                vectorizer=EmbeddingsVectorizer(self._embedder),
                distance_threshold=self.settings.prompt_cache_distance,
                ttl=self.settings.prompt_cache_ttl_s,
                filterable_fields=[{"name": "context_key", "type": "tag"}],
                redis_url=self.settings.redis_url,
            )
        return self._similar

    def lookup(self, question: str, chunk_ids: list[str], index_version: int) -> tuple[CachedAnswer, str] | None:
        """(answer, "exact" | "similar") for this prompt, or None (a MISS)."""
        try:
            cached = self.exact.lookup(prompt_text(question, chunk_ids), self._llm_string(index_version))
            if cached:
                logger.info("Redis Prompt Cache HIT (exact) for %r", question)
                return CachedAnswer.model_validate_json(cached[0].text), "exact"
            key = context_key(chunk_ids, self._llm_string(index_version))
            hits = self.similar.check(prompt=question, filter_expression=Tag("context_key") == key, num_results=1)
            if hits:
                logger.info("Redis Prompt Cache HIT (similar) for %r ~ %r", question, hits[0].get("prompt"))
                return CachedAnswer.model_validate_json(hits[0]["response"]), "similar"
        except Exception as error:  # the cache is an optimisation: failing means a MISS, never a failed search
            logger.warning("Redis Prompt Cache lookup failed, treating as a miss: %s", error)
        return None

    def store(self, question: str, chunk_ids: list[str], index_version: int, answer: CachedAnswer) -> None:
        try:
            llm_string = self._llm_string(index_version)
            self.exact.update(prompt_text(question, chunk_ids), llm_string, [Generation(text=answer.model_dump_json())])
            self.similar.store(
                prompt=question,
                response=answer.model_dump_json(),
                filters={"context_key": context_key(chunk_ids, llm_string)},
            )
        except Exception as error:
            logger.warning("Could not write to the Redis Prompt Cache: %s", error)

    def _llm_string(self, index_version: int) -> str:
        s = self.settings
        return f"{s.llm_provider}:{s.llm_model}|corpus-v{index_version}"


def prompt_text(question: str, chunk_ids: list[str]) -> str:
    """The prompt as the exact cache sees it: the question and the chunks sent with it, in rank order."""
    return f"{question.strip()}\n---\n{','.join(chunk_ids)}"


def context_key(chunk_ids: list[str], llm_string: str) -> str:
    """One tag for "these chunks, this model, this corpus version" (order ignored: the same evidence)."""
    return hashlib.sha1(f"{llm_string}|{','.join(sorted(chunk_ids))}".encode()).hexdigest()[:24]
