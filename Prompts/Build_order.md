# Build Order

This plan follows **`docs/Search_Engine_Architecture.drawio`** strictly. Every step maps to a box or arrow in the
diagram. Nothing in the diagram is dropped, merged or reordered, and nothing that is not in the diagram is added.
(`AI_Search_Engine_Architecture_v2-1.drawio` was an alternative design and is **not** used.)

**Spark Streaming was removed** (2026-09-26): measured on this project, Spark made ingestion slower (104 s vs 86 s
for the same files) and added Java, Hadoop's `winutils` on Windows and a 10 GB Docker image, with no gain on one
machine. The diagram's five "Spark Streaming:" boxes were relabelled to their plain names (TEXT Extraction, Extract
Audio & Convert to Text, CHUNKING, Text Embeddings, Image Embeddings (CLIP)); their logic is unchanged.

**The EVAL → ingestion arrow was removed** (2026-09-26): feeding eval results back into the corpus would let
earlier LLM answers be retrieved and cited as evidence (errors reinforce themselves, and they go stale when the
source documents change). Eval results are still **stored separately**, and good answers are still reused through
RESULT → FAISS Semantic Cache ("goes for caching").

## Rules

1. **The system is built in three parts, and each part is built end to end.** The parts are the three regions of the
   diagram: the **Ingestion pipeline**, the **Retrieval pipeline** and the **LLM Architecture**. A part is done
   only when every box and arrow in its region exists, is connected to its neighbouring parts, and runs from the
   CLI or the API. Nothing is left as a stub "for a later phase".
2. **The diagram decides.** If the code or this file disagrees with the diagram, the diagram wins. To change the
   architecture, change the diagram first, then this file, then the code.
3. **Each box is a plain Python function or class**, tested on its own; `pipeline.py` in each part connects them.

## Target flows (from the diagram)

**Ingestion pipeline**
```
Text Documents ──► TEXT Extraction ──► CHUNKING (Smart, Semantic) ──► Text Embeddings ──┐
PDFs/XML/CSV/JSON ─► PDFs: separate images, charts, tables │ others: just text extraction ─► CHUNKING │
                         └─ charts, images from the PDFs ─► PICTURES                                │
PICTURES ─────────────────────────────────────────► Image Embeddings (CLIP) ────────────────────────┴─► Metadata Enrichment ─► Vector Database
VIDEO ─┬─► Extract Audio & Convert to Text ─► TEXT Extraction
       └─► Takes pictures frame by frame temporarily ─► PICTURES
```

**Retrieval pipeline**
```
USER ─► QUERY ─► QUERY Enhancement ─► METADATA FILTERING ─► ELASTICSEARCH HYBRID SEARCH (KEYWORD SEARCH (BM25) + SEMANTIC SEARCH, over the Vector DB)
     ─► RERANKING (RANKFUSION) using Cross Encoder Models ─► FAISS SEMANTIC CACHE ("checks if similar question in cache") ──HIT──► USER
                                                                  │ MISS
                                                                  ▼
                                   REDIS PROMPT CACHING ("checks if exact prompt or similar prompt there or not") ──HIT──► USER
                                                                  │ MISS
                                                                  ▼
                                                            LLM ("it has its own architecture")
USER ── Sessional Queries ──► REDIS
```

**LLM Architecture**
```
LLM ─► MCP
    ─► TOOLS
    ─► GUARDRAIL ──UNSAFE──► USER ("sends the reason alongside")
             └──SAFE──► EVAL ─► RESULT ─┬─► FAISS SEMANTIC CACHE ("goes for caching")
                          │             └─► FINAL RESULT to USER
                          └─► "all the eval results are stored separately"
```

## Folder layout

One installable package, `src/search_engine/`. Each folder is a diagram region and each module is a diagram box.

| Folder | Holds |
|---|---|
| `core/` | `config.py` (pydantic-settings reading `.env`), `logging.py`, `exceptions.py` |
| `schemas/` | Pydantic models shared by all three parts (`Document`, `Chunk`, `ChunkMetadata`, query, response, eval) |
| `infra/` | Clients built once: `elasticsearch.py`, `redis.py`, `llm_client.py`, `models.py` (bge, CLIP, Whisper, cross-encoder) |
| `ingestion/` | **Ingestion pipeline**: `sources/`, `chunking.py`, `text_embed.py`, `image_embed.py`, `enrichment.py`, `vector_store.py`, `pipeline.py` |
| `retrieval/` | **Retrieval pipeline**: `query_enhance.py`, `filters.py`, `hybrid_search.py`, `rerank.py`, `semantic_cache.py`, `prompt_cache.py`, `session.py`, `pipeline.py` |
| `llm/` | **LLM Architecture**: `agent.py`, `context.py`, `mcp.py`, `tools/`, `guardrail.py`, `eval.py`, `eval_store.py`, `pipeline.py` |
| `api/` | FastAPI app, the USER entry point (`app.py`, `deps.py`, `routes/`) |
| `cli.py` | `search-engine` command: `ingest`, `search` |
| `tests/unit/` | Mirrors `src/`, no Docker needed; `-m slow` runs the real models |
| `tests/integration/` | Real Elasticsearch + Redis from `docker-compose.yml` (skipped when not running) |
| `tests/e2e/` | File in → answer out; one test per part's "Done when" |
| `eval/` | Dev-only retrieval benchmark (`corpus/`, `questions.jsonl`, `run.py`: recall@k, MRR). Not the EVAL box. |
| `data/` | Git-ignored runtime data: `faiss_cache/` (FAISS Semantic Cache), `eval_results/` |
| `docs/` | `Search_Engine_Architecture.drawio` |
| `Prompts/` | This file and the approved plans (`Prompts/plans/`) |

Entry points (`api/`, `cli.py`) only call a part's `pipeline.py`; they hold no logic.
Run things with `uv run search-engine ...` and `uv run pytest`.

---

## Part 0: Setup ✅

- `docker-compose.yml`: Elasticsearch 8.15 (basic license) + Redis (redis-stack, for the vector search that Redis
  Prompt Caching's "similar prompt" check needs)
- `core/config.py`: pydantic-settings reading `.env` (keys listed in `.env.example`)
- Check: `curl http://localhost:9200`, `docker exec search-redis redis-cli ping`

---

## Part 1: Ingestion pipeline (end to end) ✅

Diagram boxes: *Text Documents, PDFs/XML/CSV/JSON, PICTURES, VIDEO, TEXT Extraction, PDF separation of
images/charts/tables, Extract Audio & Convert to Text, Takes pictures frame by frame temporarily, CHUNKING (Smart,
Semantic), Text Embeddings, Image Embeddings (CLIP), Metadata Enrichment, Vector Database.*

### 1.1 Inputs and Text Extraction ✅
`ingestion/sources/`, one loader per diagram input, each returning a `Document` (`sections` for text,
`pictures` for images), dispatched by `load_file()`:
- `text.py`: **Text Documents** (`.txt`, `.md`), Markdown headings kept as sections
- `structured.py`: **XML, CSV, JSON**, "just text extraction", one `field: value` section per record
- `pdf.py`: **PDFs, separating images, charts, tables**: page text → Chunking; tables → Markdown sections
  (`kind="table"`) → Chunking; **charts and images from the PDFs → PICTURES** (`Document.pictures`)
- `image.py`: **PICTURES** (`.png`, `.jpg`, `.webp`, …) → Image Embeddings
- `video.py`: **VIDEO** splits two ways:
  - **Extract Audio & Convert to Text** (`extract_audio_to_text()`, faster-whisper) → timestamped transcript
    sections joining Text Extraction (`modality="video_transcript"`)
  - **Takes pictures frame by frame temporarily** (`sample_frames()`, PyAV) → timestamped frames → PICTURES
    (`modality="video_frame"`); frames are held in memory only and never written to disk

### 1.2 Chunking (Smart, Semantic) ✅
`ingestion/chunking.py`, `chunk_document(doc, embeddings, settings)`:
- *Semantic* for prose (txt, Markdown, PDF text): cut where the meaning shifts (our own splitter on bge);
  oversized pieces re-split
- *Smart* for records (whole records packed, oversized records split at field lines with their first line
  repeated), tables (split between rows, header repeated) and transcripts (kept one-to-one with timestamps)
- chunks never cross a page, heading, transcript section or table edge

### 1.3 Text Embeddings and Image Embeddings (CLIP) ✅
- `ingestion/text_embed.py`: **Text Embeddings** (sentence-transformers, `BAAI/bge-small-en-v1.5`, 384-dim) →
  `Chunk.text_embedding`; also the LangChain `Embeddings` used by semantic chunking and query embedding
- `ingestion/image_embed.py`: **Image Embeddings (CLIP)** (sentence-transformers `clip-ViT-B-32`, 512-dim):
  each `Picture` → a `Chunk` with `image_embedding`; `ClipTextEmbeddings` is CLIP's text encoder as a LangChain
  `Embeddings`, for text-to-image search
- models loaded once in `infra/models.py`

### 1.4 Metadata Enrichment ✅
`ingestion/enrichment.py`, `enrich(doc, chunks)`, after both embedding boxes: a stable chunk id
(`<doc_id>-<chunk_index>`, `doc_id` = hash of the file path, so re-ingesting replaces instead of duplicating) and
the typed `ChunkMetadata` fields (`schemas/chunk.py`): file name/type, content, title, section, author, created,
file_modified, ingested_at, page_count, duration_s, language; `modality`, `page` and `timestamp` stay top-level
Chunk fields. No LLM calls.

### 1.5 Vector Database ✅
`ingestion/vector_store.py`, through **LangChain** (`langchain-elasticsearch` `ElasticsearchStore`): two stores in
the same Elasticsearch, because a LangChain store manages one vector field per index:
- text store, index `search_chunks_text`: text, table, record and transcript chunks, `text_embedding` (384, bge)
- image store, index `search_chunks_images`: image, chart and frame chunks, `image_embedding` (512, CLIP)

LangChain creates each index (`DenseVectorStrategy`, `metadata_mappings` generated from `ChunkMetadata`, English
default analyzer for the BM25 `text` field) and does every write (`add_embeddings` with the precomputed vectors).
Documents are `{text, <vector>, metadata: {ChunkMetadata fields + source, modality, page, timestamp}}`
(`source`, the file path, added in Part 2 for citations: re-ingest older files to fill it). Re-ingesting a file
deletes its old chunks by `doc_id` first. Each write bumps the Redis counter `search:index_version`, so cached
answers can be dropped when the corpus changes.

### 1.6 Pipeline ✅
- `ingestion/pipeline.py`: `IngestionPipeline` connects 1.1 → 1.5 for one file or a folder; one bad file is
  recorded as failed and doesn't stop the rest ✅
- `search-engine ingest <path>` ✅ (4 real files → 70 chunks in Elasticsearch: text, tables, records, images,
  video transcript and frames)
- `tests/e2e/test_ingestion_done_when.py`: one file of each of the 8 input types → real Elasticsearch, every
  chunk kind and modality present, embedded, enriched, with page / timestamp citations
  (`uv run pytest -m "slow and integration"`)

**Done when:** running `search-engine ingest` on a txt/md/pdf/xml/csv/json file, an image and a video produces
enriched, embedded chunks in Elasticsearch: text chunks, PDF tables, PDF charts and images, video transcript chunks
and video frames, each with the right `modality`.

---

## Part 2: Retrieval pipeline (end to end) ✅

Diagram boxes: *USER → QUERY, QUERY Enhancement, METADATA FILTERING, ELASTICSEARCH HYBRID SEARCH (KEYWORD SEARCH
(BM25) + SEMANTIC SEARCH), RERANKING (RANKFUSION) using Cross Encoder Models, FAISS SEMANTIC CACHE (HIT/MISS),
REDIS PROMPT CACHING (HIT/MISS), Sessional Queries → REDIS, LLM.*

1. `schemas/query.py`, `schemas/response.py`: request/response models (query, session id, filters; answer,
   citations, cache source, guardrail reason) ✅
2. `retrieval/query_enhance.py`: **QUERY Enhancement**: one fast LLM call (free API: Groq by default, Gemini through
   LangChain `init_chat_model`; Groq `openai/gpt-oss-20b`, JSON-schema structured output) that rewrites the query
   to stand alone, adds keywords and extracts filters; if the LLM fails, the query is searched as typed ✅
3. `retrieval/filters.py`: **METADATA FILTERING**, which turns the query or user options into filters on the
   Part 1.4 fields (modality, content, file type, file name, created date); the user's options win over the
   query's ✅
4. `retrieval/hybrid_search.py`: **ELASTICSEARCH HYBRID SEARCH** over the Vector Database, through
   **LangChain's built-in hybrid search** on the two 1.5 stores:
   - text store with `DenseVectorStrategy(hybrid=True, rrf=False)`: **KEYWORD SEARCH (BM25)** on `text` +
     **SEMANTIC SEARCH** (kNN on `text_embedding`, bge query embedding) in one Elasticsearch query
   - image store: kNN on `image_embedding` (CLIP text encoder), so text queries find PDF charts, images and video
     frames
   - `rrf=False` because ES's own RRF returns **403** on the basic license ("non-compliant for Reciprocal Rank
     Fusion"); rank fusion is done in the next step instead
   - a store the filters rule out (e.g. only charts) or whose index doesn't exist yet is skipped ✅
5. `retrieval/rerank.py`: **RERANKING (RANKFUSION) using Cross Encoder Models**: rank fusion (RRF via LangChain's
   `EnsembleRetriever`, from `langchain-classic`, merged by chunk id) of the text-hybrid and image-kNN lists, then
   a **Cross Encoder** (LangChain `CrossEncoderReranker`, local `cross-encoder/ms-marco-MiniLM-L6-v2`, chosen over
   `bge-reranker-base` on the eval/ numbers: same or better, 3.5x faster, 1/14 the memory) over the fused text hits.
   The cross encoder reads text only, so up to `rerank_image_slots` (2) image hits follow the text hits; when
   only pictures match (e.g. only charts asked for) they fill the list. Images go after text because CLIP's
   text-to-image scores barely separate matching from unrelated pictures (measured: 0.637 for the matching chart,
   0.645 for an unrelated query), so no score cut-off works ✅
6. The cache boxes, **after Reranking and before the LLM**, in the diagram's order ✅
   - `retrieval/semantic_cache.py`, **FAISS SEMANTIC CACHE**: "checks if similar question in cache". LangChain
     `FAISS` on the bge embedding of the rewritten question; HIT when cosine ≥ 0.92 with the same filters. The
     cache belongs to one corpus version (`search:index_version`): a new ingest empties it. Saved in
     `data/faiss_cache/`. HIT → USER, MISS → Redis.
   - `retrieval/prompt_cache.py`, **REDIS PROMPT CACHING**: "checks if exact prompt or similar prompt there or not".
     Prompt = rewritten question + reranked chunk ids. Exact = LangChain `RedisCache`; similar = `redisvl`
     `SemanticCache` (the library under langchain-redis), a similar question **with the same chunks** (tag filter;
     langchain-redis' own class only checks the single nearest entry). Model + corpus version in every key, 24 h
     TTL. HIT → USER, MISS → LLM.
   - `retrieval/session.py`, **Sessional Queries**: `RedisChatMessageHistory` (langchain-community; the
     langchain-redis class is deprecated and fails with redis-py 8), per session id, 1 h TTL. Feeds Query
     Enhancement and the LLM.
   - Only LLM answers with citations are cached. A cache error is a MISS; without the index version the caches
     are skipped.
7. **LLM** box: `llm/agent.py` `LLMAgent.answer()`, a direct call with the numbered chunks (file, page / video
   time, content; pictures by their description only), citing [n] (Groq `openai/gpt-oss-120b`; its `【n】`
   citations are rewritten to [n]). `retrieval/pipeline.py` `SearchPipeline`
   connects every box in the diagram's order; citations are the chunks the answer cites. An LLM failure
   (Gemini 503 / 429) raises `LLMUnavailableError`. Part 3 replaces the inside of `answer()` ✅
8. `api/`: FastAPI `POST /search`, `GET /health`; models loaded and warmed at startup
   (`uv run uvicorn search_engine.api.app:app`). CLI: `search-engine search "<query>" [--session] [--file-type]
   [--file-name] [--modality] [--content] [--top-k]` ✅
9. `eval/`: 25 questions over `eval/corpus/` (+ a generated PDF with a table and a chart, and two photos);
   `uv run python -m eval.run [--enhance] [--rerank-model M]` prints recall@1/5/10, MRR and time per stage (text
   hybrid → + image search, RRF → + cross encoder). Retrieval only: no LLM tokens unless `--enhance` ✅

**Done when:** a question returns a cited answer produced from reranked hybrid results (text and image hits), and
asking the same or a similar question again is answered from the FAISS or Redis cache without calling the LLM. ✅
(`tests/e2e/test_retrieval_done_when.py`; real run with Gemini through the CLI and the API)

---

## Part 3: LLM Architecture (end to end) ✅

Diagram boxes: *LLM, MCP, TOOLS, GUARDRAIL (SAFE / UNSAFE, "sends the reason alongside"), EVAL, RESULT,
"goes for caching", FINAL RESULT to USER, "all the eval results are stored separately".*

**Two LLM providers are used together** (`infra/llm_client.py`; `.env`: `GROQ_API_KEY`, `GROQ_MODEL`,
`GROQ_FAST_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`), so neither free rate limit is reached quickly. Each role
prefers one provider and falls back to the other on an error, rate limit or unusable output:

| Role | Used by | First choice | Falls back to |
|---|---|---|---|
| fast | Query Enhancement | Groq `openai/gpt-oss-20b` | Gemini |
| answer | the LLM (agent) | Groq `openai/gpt-oss-120b` | Gemini |
| guard | Guardrail | Groq `openai/gpt-oss-safeguard-20b` | Gemini |
| judge | Eval | Gemini `gemini-2.5-flash` | Groq `openai/gpt-oss-20b` |

1. `llm/agent.py`: the **LLM** as an agent (LangChain `create_agent`, on LangGraph): answers from the numbered
   passages and cites [n]; can call tools up to `agent_max_tool_calls` (4); if the agent run fails, one direct LLM
   call answers from the passages in hand ✅
   - `llm/tools/`: **TOOLS**, one file per tool: `search_documents` (hybrid search + rerank again),
     `read_surrounding` (the chunks before and after a passage), `list_files`, `calculator` (parsed, never
     executed). Passages a tool finds continue the numbering (`llm/context.py`), so they can be cited ✅
   - `llm/mcp.py`: **MCP** client (`langchain-mcp-adapters` `MultiServerMCPClient`): tools of the servers in
     `MCP_SERVERS` (.env; none by default) are offered to the agent as `<server>_<tool>`; a server that can't be
     reached is skipped ✅
2. `llm/guardrail.py`: **GUARDRAIL** on the LLM output: rules for credentials (no LLM call), then a safety model
   judging the answer against a written policy (harmful, secrets, sensitive personal data, prompt injection) ✅
   - **UNSAFE** → back to the USER with the reason (`guardrail_reason`, source `guardrail_blocked`); not cached,
     EVAL skipped
   - **SAFE** → EVAL
   - safety model unreachable → the answer passes on the rules only and is never cached
3. `llm/eval.py`: **EVAL** (LLM-as-judge, JSON-schema output): faithfulness, relevance, citation correctness,
   1–5, judged against the passages and the tool results; passes when every score ≥ `eval_pass_score` (4). The
   user gets the answer with its scores either way ✅
4. `llm/pipeline.py`: `LLMPipeline.run()` builds the **RESULT** (LLM → GUARDRAIL → EVAL), which ✅
   - **goes for caching**: written to the FAISS Semantic Cache and the Redis Prompt Cache only when it passed
     GUARDRAIL + EVAL (and cites something)
   - is returned as the **FINAL RESULT to USER** (answer, citations, eval scores, tools used)
5. `llm/eval_store.py`: **"all the eval results are stored separately"**: one JSON line per LLM answer in
   `data/eval_results/<date>.jsonl` (question, answer, passages, tools, guardrail verdict, eval scores). It is
   **never** ingested: Ingestion skips that folder, so LLM answers can't come back as evidence ✅
6. `retrieval/pipeline.py` calls `LLMPipeline.run()` in place of Part 2's direct LLM call ✅

LLM calls per new question: Query Enhancement + agent (1 + one per tool call) + guardrail + eval = 4 or more,
split over the two providers. A cache hit: 1.

**Done when:** every answer passes through LLM (with MCP/TOOLS) → GUARDRAIL → EVAL → RESULT; unsafe answers come
back with a reason; safe results are cached; and every eval result is stored separately (and does not appear in
search results). ✅ (`tests/e2e/test_llm_done_when.py`; real run on Groq + Gemini through the full pipeline)

---

## Implementation notes (these don't change the architecture)

- **Vector Database = the Elasticsearch indexes**, accessed through `langchain-elasticsearch`. The diagram feeds the
  Vector Database into Elasticsearch Hybrid Search, so the same ES serves both boxes.
- **CLIP and text embeddings are different vector spaces.** Keep them in separate indexes and fuse only their
  rankings (rank fusion), never the raw scores.
- **Eval results stay out of the corpus.** Only original documents are retrievable; eval results live in
  `data/eval_results/` and good answers are reused through the FAISS Semantic Cache.
- **Scaling ingestion later** (many files, uploads): a small worker on a Redis job queue calling
  `IngestionPipeline.ingest_file`, and a GPU for Whisper and the embedding models. Add to the diagram first.

## LLM call budget

LLM calls per query, following the diagram as it is. Embedding models, CLIP, Whisper and the cross-encoder are
local models, not LLM calls.

| Box | LLM calls | Notes |
|---|---|---|
| Query Enhancement | 1 | Rewrite, expansion or HyDE. Runs on every query, including cache hits. |
| Metadata Filtering | 0 (or 1) | 0 with rule-based or user-supplied filters; 1 if an LLM extracts filters from the query. |
| Hybrid Search, Reranking | 0 | |
| FAISS Semantic Cache, Redis Prompt Caching | 0 | Embedding lookups only. |
| LLM | 1 + N | N = number of TOOLS / MCP round-trips. |
| GUARDRAIL | 1 (or 0) | 0 if a small safety classifier is used instead of an LLM. |
| EVAL | 1 | LLM-as-judge. |

- **Cache hit:** 1 call (Query Enhancement). The caches sit after Reranking, so a hit still pays for enhancement
  and the full search.
- **Cache miss:** about 4 + N calls, or 3 + N with a classifier guardrail.
- **Ingestion:** 0 calls.

## Open decisions

- **Kafka / Spark.** Not in the diagram. Ingestion runs through `IngestionPipeline` directly.
