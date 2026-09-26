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
| `infra/` | Clients built once: `elasticsearch.py`, `redis.py`, `faiss_index.py`, `llm_client.py`, `models.py` (bge, CLIP, Whisper, cross-encoder) |
| `ingestion/` | **Ingestion pipeline**: `sources/`, `chunking.py`, `text_embed.py`, `image_embed.py`, `enrichment.py`, `vector_store.py`, `pipeline.py` |
| `retrieval/` | **Retrieval pipeline**: `query_enhance.py`, `filters.py`, `hybrid_search.py`, `rerank.py`, `semantic_cache.py`, `prompt_cache.py`, `session.py`, `pipeline.py` |
| `llm/` | **LLM Architecture**: `agent.py`, `mcp.py`, `tools/`, `guardrail.py`, `eval.py`, `eval_store.py`, `pipeline.py` |
| `api/` | FastAPI app, the USER entry point (`app.py`, `deps.py`, `routes/`) |
| `cli.py` | `search-engine` command: `ingest`, `search` |
| `tests/unit/` | Mirrors `src/`, no Docker needed; `-m slow` runs the real models |
| `tests/integration/` | Real Elasticsearch + Redis from `docker-compose.yml` (skipped when not running) |
| `tests/e2e/` | File in → answer out; one test per part's "Done when" |
| `eval/` | Dev-only retrieval benchmark (recall@k, MRR). Not the EVAL box. |
| `data/` | Git-ignored runtime data: `eval_results/` |
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
Documents are `{text, <vector>, metadata: {ChunkMetadata fields + modality, page, timestamp}}`. Re-ingesting a file
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

## Part 2: Retrieval pipeline (end to end)

Diagram boxes: *USER → QUERY, QUERY Enhancement, METADATA FILTERING, ELASTICSEARCH HYBRID SEARCH (KEYWORD SEARCH
(BM25) + SEMANTIC SEARCH), RERANKING (RANKFUSION) using Cross Encoder Models, FAISS SEMANTIC CACHE (HIT/MISS),
REDIS PROMPT CACHING (HIT/MISS), Sessional Queries → REDIS, LLM.*

1. `schemas/query.py`, `schemas/response.py`: request/response models (query, session id, filters; answer,
   citations, cache source, guardrail reason).
2. `retrieval/query_enhance.py`: **QUERY Enhancement** (LLM rewrite, expansion or HyDE).
3. `retrieval/filters.py`: **METADATA FILTERING**, which turns the query or user options into filters on the
   Part 1.4 fields (modality, file type, source, dates…).
4. `retrieval/hybrid_search.py`: **ELASTICSEARCH HYBRID SEARCH** over the Vector Database, all through
   **LangChain** (the two 1.5 stores + a LangChain BM25 retriever):
   - **KEYWORD SEARCH (BM25)** on `text`
   - **SEMANTIC SEARCH**: kNN on `text_embedding` (bge query embedding), plus kNN on `image_embedding` (CLIP text
     encoder) so text queries find PDF charts, images and video frames
   - LangChain's built-in `DenseVectorStrategy(hybrid=True)` was tested on the local ES and returns **403** (basic
     license: "non-compliant for Reciprocal Rank Fusion"), so the lists are fused in the next step instead
5. `retrieval/rerank.py`: **RERANKING (RANKFUSION) using Cross Encoder Models**: rank fusion (RRF via LangChain's
   `EnsembleRetriever`, from `langchain-classic`) of the BM25, text-kNN and image-kNN lists, then a **Cross
   Encoder** (`BAAI/bge-reranker-base`, local) over the fused top-N.
6. The cache boxes, **after Reranking and before the LLM**, in the diagram's order:
   - `retrieval/semantic_cache.py`, **FAISS SEMANTIC CACHE**: "checks if similar question in cache". HIT → USER,
     MISS → Redis.
   - `retrieval/prompt_cache.py`, **REDIS PROMPT CACHING**: "checks if exact prompt or similar prompt there or not".
     Exact = hash of the normalized prompt; similar = Redis vector search over prompt embeddings. HIT → USER,
     MISS → LLM.
   - `retrieval/session.py`, **Sessional Queries**: the USER's queries in the current session, stored in Redis
     (keyed by session id, with a TTL). They feed the prompt cache and give the LLM conversation context.
7. **LLM** box: `retrieval/pipeline.py` hands the prompt to `llm/agent.py`. In this part that is a direct LLM call
   with the reranked chunks as context. Part 3 replaces it with the full LLM Architecture behind the same function.
8. `api/routes/search.py`: FastAPI `/search` (USER → QUERY → answer), and `search-engine search "<query>"`.
9. `eval/`: 20–30 questions with known answers to measure recall@k and MRR while tuning (a dev tool, not the EVAL box).

**Done when:** a question returns a cited answer produced from reranked hybrid results (text and image hits), and
asking the same or a similar question again is answered from the FAISS or Redis cache without calling the LLM.

---

## Part 3: LLM Architecture (end to end)

Diagram boxes: *LLM, MCP, TOOLS, GUARDRAIL (SAFE / UNSAFE, "sends the reason alongside"), EVAL, RESULT,
"goes for caching", FINAL RESULT to USER, "all the eval results are stored separately".*

1. `llm/agent.py`: the **LLM** as an agent that can call:
   - `llm/mcp.py`: **MCP** client (connects to MCP servers)
   - `llm/tools/`: **TOOLS** (local function tools, one file per tool)
2. `llm/guardrail.py`: **GUARDRAIL** on the LLM output:
   - **UNSAFE** → back to the USER with the reason
   - **SAFE** → EVAL
3. `llm/eval.py`: **EVAL** (LLM-as-judge: faithfulness to the context, relevance, citation correctness), as JSON.
4. `llm/pipeline.py`: builds the **RESULT**, which:
   - **goes for caching**: written to the FAISS Semantic Cache (only results that passed GUARDRAIL + EVAL)
   - is returned as the **FINAL RESULT to USER**
5. `llm/eval_store.py`: **"all the eval results are stored separately"**: every eval result is saved as a JSON
   record in its own store (`data/eval_results/`), for monitoring answer quality and tuning retrieval. It is
   **never** ingested into the Vector Database, so LLM answers can't come back as evidence.
6. Swap the direct LLM call from Part 2 for this pipeline behind the same function.

**Done when:** every answer passes through LLM (with MCP/TOOLS) → GUARDRAIL → EVAL → RESULT; unsafe answers come
back with a reason; safe results are cached; and every eval result is stored separately (and does not appear in
search results).

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
