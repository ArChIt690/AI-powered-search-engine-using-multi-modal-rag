# Part 2 Retrieval pipeline: plan

> Approved plan (2026-09-28). The user chose a free LLM API (Groq, or Gemini through the same config) and
> LangChain's built-in hybrid search (`DenseVectorStrategy(hybrid=True)`) instead of a separate BM25 retriever.

## Flow (v1 diagram)
```
USER → QUERY → Query Enhancement → Metadata Filtering → ES Hybrid Search (text hybrid + image kNN)
     → Reranking (RRF → Cross Encoder) → FAISS Semantic Cache ──HIT──► USER
                                           │ MISS
                                        Redis Prompt Cache ──HIT──► USER
                                           │ MISS
                                          LLM → cited answer → USER
USER ── Sessional Queries ──► Redis
```

## Boxes

| # | Box | File | Approach | Library |
|---|---|---|---|---|
| 1 | QUERY | `schemas/query.py`, `schemas/response.py` | Request: query, session id, filters. Response: answer, citations (file, page, timestamp), source (`faiss` / `redis` / `llm`) | pydantic |
| 2 | Query Enhancement | `retrieval/query_enhance.py` | One fast LLM call, structured output: the query rewritten to stand alone (using session history), extra keywords, filters the query mentions | LangChain `init_chat_model` + `with_structured_output` |
| 3 | Metadata Filtering | `retrieval/filters.py` | User filters + LLM filters → ES `filter` clauses on the 1.4 fields; field names checked against `ChunkMetadata` | plain Python |
| 4 | ES Hybrid Search | `retrieval/hybrid_search.py` | Text: the 1.5 text store with LangChain's **`DenseVectorStrategy(hybrid=True, rrf=False)`**: BM25 + bge kNN in one ES query. Images: the 1.5 image store, kNN with the CLIP text encoder. Same filters on both | `langchain-elasticsearch` |
| 5 | Reranking (RankFusion + Cross Encoder) | `retrieval/rerank.py` | RRF of the two lists (`EnsembleRetriever`), then `BAAI/bge-reranker-base` over the fused top 30 → top 8; the top image hits keep their RRF place (the cross encoder only reads text) | `langchain-classic`, `langchain-community` `HuggingFaceCrossEncoder` |
| 6a | FAISS Semantic Cache | `retrieval/semantic_cache.py` | Similar question (bge, cosine ≥ ~0.92) → cached answer; `data/faiss_cache/` | LangChain `FAISS`, `faiss-cpu` |
| 6b | Redis Prompt Caching | `retrieval/prompt_cache.py` | Exact (hash) or similar (vector search) full prompt → cached answer | `langchain-redis` `RedisCache`, `RedisSemanticCache` |
| 6c | Sessional Queries | `retrieval/session.py` | Session history in Redis, 1 h TTL; feeds Query Enhancement and the LLM | `langchain-redis` `RedisChatMessageHistory` |
| 7 | LLM | `retrieval/pipeline.py` → `llm/agent.py` | Direct call for now, numbered chunks as context, answer cites `[1]`; Part 3 replaces it behind the same function | `init_chat_model` |
| 8 | USER entry | `api/routes/search.py`, `cli.py` | `POST /search`, `search-engine search "<query>"` | FastAPI, argparse |
| 9 | Dev benchmark | `eval/` | 20–30 questions, recall@k, MRR | plain Python |

## Decisions
- **Hybrid search is LangChain's built-in class.** Checked on the local ES (basic license): `hybrid=True` with
  `rrf=True` → 403 ("non-compliant for RRF"); with `rrf=False` → works (ES sums the BM25 and kNN scores). Rank
  fusion then happens in Reranking. `similarity_search_with_score` is not supported in hybrid mode, so hybrid
  results are used by rank only.
- **The cross encoder reads text only**, so image hits (text = a label) keep their RRF place instead of being
  pushed down by it.
- **Cache entries store `search:index_version`**; after a new ingest they count as a miss.
- **The LLM sees image hits as label + location** (file, page, timestamp), not pixels: 1.5 stores no image bytes.
- **LLM: free API, switchable in `.env`** through `init_chat_model`: `LLM_PROVIDER=groq` (default,
  `llama-3.3-70b-versatile`, fast model `llama-3.1-8b-instant`) or `google_genai` (`gemini-2.5-flash`).
- **LLM calls per question:** cache hit 1 (Query Enhancement runs before the caches in the diagram), miss 2.

## Tests
- Unit per box (fake LLM, fake stores); integration on real ES + Redis.
- E2E "Done when": ingest the 8-file corpus, ask → cited answer with text and image hits; ask again → from cache,
  no LLM call.
