# Build Order

This plan follows `Search_Engine_Architecture.drawio` strictly. Every step maps to a box or arrow in the diagram.
Nothing in the diagram is dropped, merged or reordered, and nothing that is not in the diagram is added.

## Rules

1. **The system is built in three parts, and each part is built end to end.** The parts are the three regions of the
   diagram: the **Ingestion pipeline**, the **Retrieval pipeline** and the **LLM Architecture**. A part is done
   only when every box and arrow in its region exists, is connected to its neighbouring parts, and runs from the
   CLI or the API. Nothing is left as a stub "for a later phase".
2. **The diagram decides.** If the code or this file disagrees with the diagram, the diagram wins. To change the
   architecture, change the diagram first, then this file, then the code.
3. **Every bold "Spark Streaming:" box is a Spark job.** Each box's logic is written as a plain Python function
   (so it can be tested without Spark), and the Spark job in `streaming/jobs/` calls that function.
   Both are built in the same part.

## Target flows (from the diagram)

**Ingestion pipeline**
```
Text Documents ──► Text Extraction ──► Chunking (Smart, Semantic) ──► Text Embeddings ──┐
PDFs/XML/CSV/JSON ─► PDFs: separate images, charts, tables │ others: text only ─► Chunking │
                         └─ charts, images from PDFs ─► PICTURES                         ├─► Metadata Enrichment ─► Vector Database
PICTURES ──────────────────────────────────────────► Image Embeddings (CLIP) ────────────┘
VIDEO ─┬─► Extract Audio & Convert to Text ─► Text Extraction (text path)
       └─► Takes pictures frame by frame (temporary) ─► PICTURES
```

**Retrieval pipeline**
```
USER ─► QUERY ─► Query Enhancement ─► Metadata Filtering ─► Elasticsearch Hybrid Search (BM25 + Semantic, over the Vector DB)
     ─► Reranking (Rank Fusion + Cross Encoder) ─► FAISS Semantic Cache ──HIT──► USER
                                                        │ MISS
                                                        ▼
                                  Redis Prompt Caching (exact or similar prompt) ──HIT──► USER
                                                        │ MISS
                                                        ▼
                                                  LLM (LLM Architecture)
USER ── Sessional Queries ──► Redis
```

**LLM architecture**
```
LLM ─► MCP
    ─► TOOLS
    ─► GUARDRAIL ──UNSAFE──► USER (sends the reason alongside)
             └──SAFE──► EVAL ─► RESULT ─┬─► FAISS Semantic Cache ("goes for caching")
                          │             └─► FINAL RESULT to USER
                          └─► eval results stored separately (JSON) ─► back into Ingestion (PDFs/XML/CSV/JSON input)
```

## Folder layout

One installable package, `src/search_engine/`. Each folder is a diagram region and each module is a diagram box.
Paths below are relative to `src/search_engine/` unless they start with a top-level folder.

| Folder | Holds |
|---|---|
| `core/` | `config.py` (pydantic-settings reading `.env`), `logging.py`, `exceptions.py` |
| `schemas/` | Pydantic models shared by all three parts (`Chunk`, `Document`, query, response, eval) |
| `infra/` | Clients for external services, built in one place and passed into the boxes: Elasticsearch, Redis, FAISS, the LLM provider (`llm_client.py`), and model loading (`models.py`: embeddings, CLIP, Whisper, cross-encoder) |
| `ingestion/` | **Ingestion pipeline** |
| `retrieval/` | **Retrieval pipeline** |
| `llm/` | **LLM Architecture** |
| `streaming/` | `spark.py` (SparkSession builder) and `jobs/`: one Spark job per bold "Spark Streaming:" box |
| `api/` | FastAPI app, the USER entry point (`app.py`, `deps.py`, `routes/`) |
| `cli.py` | `search-engine` command: `ingest`, `stream`, `search` |
| `tests/unit/` | Mirrors `src/`, no Docker needed |
| `tests/integration/` | Real Elasticsearch + Redis from `docker-compose.yml` |
| `tests/e2e/` | File in → answer out; one test per part's "Done when" |
| `eval/` | Dev-only retrieval benchmark (recall@k, MRR). Not the EVAL box. |
| `data/` | Git-ignored runtime data: `landing/` (stream source), `eval_results/` |
| `docker/` | Dockerfiles (`Dockerfile.api`, `Dockerfile.spark`) |
| `docs/` | This file and `Search_Engine_Architecture.drawio` |

Entry points (`api/`, `cli.py`, `streaming/jobs/`) only call a part's `pipeline.py`; they hold no logic.
Run things with `uv run search-engine ...` and `uv run pytest`.

---

## Part 0: Setup ✅

- `docker-compose.yml`: Elasticsearch + Redis (redis-stack, for vector search in the prompt cache)
- `core/config.py`: pydantic-settings reading `.env` (keys listed in `.env.example`)
- Check: `curl http://localhost:9200`, `docker exec search-redis redis-cli ping`

---

## Part 1: Ingestion pipeline (end to end)

Status: **to rebuild in `ingestion/`.** The first version (text path only) lived in `Search_Engine/Data/` and was
removed in commit `ac89dbb`. `cli.py ingest` already imports `ingestion.pipeline.IngestionPipeline`, which this
part creates.

Diagram boxes: *Text Documents, PDFs/XML/CSV/JSON, PICTURES, VIDEO, Spark Streaming: Text Extraction,
PDF separation of images/charts/tables, Spark Streaming: Extract Audio & Convert to Text, Takes pictures frame by
frame temporarily, Spark Streaming: Chunking (Smart, Semantic), Spark Streaming: Text Embeddings,
Spark Streaming: Image Embeddings (CLIP), Metadata Enrichment, Vector Database.*

### 1.1 Inputs and Text Extraction
`ingestion/sources/`, one loader per diagram input, each returning a `Document`:
- `text.py`: **Text Documents** (`.txt`, `.md`), keeping Markdown headings as sections
- `structured.py`: **XML, CSV, JSON**, "for others, just text extraction" (one section per record)
- `pdf.py`: **PDFs, separating images, charts, tables**:
  - text → Chunking
  - tables → serialized to Markdown text → Chunking
  - **charts, images from the PDFs → PICTURES** (the arrow in the diagram)
- `image.py`: **PICTURES** (`.png`, `.jpg`, …) → Image Embeddings
- `video.py`: **VIDEO** splits two ways:
  - **Extract Audio & Convert to Text** (ffmpeg + Whisper) → the transcript joins **Text Extraction**
    (`modality="video_transcript"`, with timestamps)
  - **Takes pictures frame by frame, temporarily** (keyframe sampling into a temp dir) → **PICTURES**
    (`modality="video_frame"`, with timestamps). Delete the temp frames after embedding.

### 1.2 Chunking (Smart, Semantic)
`ingestion/chunking.py`, with both parts of the box:
- *Smart*: structure-aware splitting (headings, paragraphs, table rows, CSV rows, JSON objects) with overlap
- *Semantic*: split where the embedding similarity between neighbouring sentences drops
- `chunking_strategy="auto"`: smart for structured files, semantic for prose

### 1.3 Text Embeddings and Image Embeddings (CLIP)
- `ingestion/text_embed.py`: **Text Embeddings** (sentence-transformers, `BAAI/bge-small-en-v1.5`) → `text_embedding`
- `ingestion/image_embed.py`: **Image Embeddings (CLIP)** (ViT-B/32) → `image_embedding`

### 1.4 Metadata Enrichment
`ingestion/enrichment.py`, which runs *after* both embedding boxes, as in the diagram: chunk id, source, file
type, modality, page, timestamp, dates, title/section. Metadata Filtering in Part 2 depends on these fields.

### 1.5 Vector Database
`ingestion/vector_store.py`: one Elasticsearch index through LangChain's `ElasticsearchStore`, with a `text` field
(BM25, English analyzer), `text_embedding` and `image_embedding` `dense_vector` fields, and typed `metadata.*`
fields. A second `ElasticsearchStore` with `vector_query_field="image_embedding"` writes the CLIP vectors.

### 1.6 Pipeline and Spark Streaming
- `ingestion/pipeline.py`: `IngestionPipeline` connects 1.1 → 1.5 for one file or a folder
- `streaming/jobs/`: one job per bold box (**Text Extraction, Chunking, Text Embeddings,
  Image Embeddings (CLIP), Extract Audio & Convert to Text**), each calling the functions above
- Stream source: a watched landing folder, `data/landing/` (Spark file source), so new files, including the eval
  JSON from Part 3, are picked up continuously
- Add Spark to `docker-compose.yml`
- `search-engine ingest <path>` for a one-off run, `search-engine stream` to start the Spark jobs

**Done when:** dropping a txt/md/pdf/xml/csv/json file, an image or a video into the landing folder (or running
`search-engine ingest`) produces enriched, embedded chunks in Elasticsearch: text chunks, PDF tables, PDF charts and
images, video transcript chunks and video frames, each with the right `modality`.

---

## Part 2: Retrieval pipeline (end to end)

Diagram boxes: *USER → QUERY, Query Enhancement, Metadata Filtering, Elasticsearch Hybrid Search
(Keyword Search BM25 + Semantic Search), Reranking (Rank Fusion using Cross Encoder models), FAISS Semantic Cache
(HIT/MISS), Redis Prompt Caching (HIT/MISS), Sessional Queries, LLM.*

1. `schemas/query.py`, `schemas/response.py`: request/response models (query, session id, filters; answer,
   citations, cache source, guardrail reason).
2. `retrieval/query_enhance.py`: **Query Enhancement** (LLM rewrite, expansion or HyDE).
3. `retrieval/filters.py`: **Metadata Filtering**, which turns the query or user options into ES `filter` clauses
   on the fields from Metadata Enrichment (modality, file type, source, dates…).
4. `retrieval/hybrid_search.py`: **Elasticsearch Hybrid Search** over the Vector Database, with both sub-searches:
   - **Keyword Search (BM25)** on `text`
   - **Semantic Search**: kNN on `text_embedding`, plus kNN on `image_embedding` using the CLIP text encoder on
     the query, so text queries can find PDF charts, images and video frames
   - Elasticsearch's built-in RRF needs a paid license (the local basic license returns 403), so the lists are
     fused in the next step.
5. `retrieval/rerank.py`: **Reranking**. First **Rank Fusion** (RRF, in our code) of the BM25, text-kNN and
   image-kNN lists, then a **Cross Encoder** (`BAAI/bge-reranker-base`) over the fused top-N.
6. The three cache boxes, one module each, **after Reranking and before the LLM**, in the diagram's order:
   - `retrieval/semantic_cache.py`, **FAISS Semantic Cache**: "checks if a similar question is in the cache". HIT → USER, MISS → Redis.
   - `retrieval/prompt_cache.py`, **Redis Prompt Caching**: "checks if the exact prompt or a similar prompt is there". Exact = hash of the
     normalized prompt; similar = Redis vector search over prompt embeddings. HIT → USER, MISS → LLM.
   - `retrieval/session.py`, **Sessional Queries**: the USER's queries in the current session, stored in Redis (keyed by session id,
     with a TTL). They feed the prompt cache and give the LLM conversation context.
7. **LLM** box: `retrieval/pipeline.py` hands the prompt to `llm/agent.py`. In this part that is a direct LLM call
   with the reranked chunks as context. Part 3 replaces it with the full LLM Architecture behind the same function.
8. `api/routes/search.py`: FastAPI `/search` (USER → QUERY → answer), and `search-engine search "<query>"`.
9. `eval/`: 20–30 questions with known answers to measure recall@k and MRR while tuning (a dev tool, not the EVAL box).

**Done when:** a question returns a cited answer produced from reranked hybrid results (text and image hits), and
asking the same or a similar question again is answered from the FAISS or Redis cache without calling the LLM.

---

## Part 3: LLM Architecture (end to end)

Diagram boxes: *LLM, MCP, TOOLS, GUARDRAIL (SAFE / UNSAFE, "sends the reason alongside"), EVAL, RESULT,
"goes for caching", FINAL RESULT to USER, "all the eval results are stored separately", "the EVAL result goes in
a JSON format to the RAG pipeline for context".*

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
5. **Eval feedback loop**:
   - every eval result is saved as a JSON record in its own store, **separately** (`llm/eval_store.py` → `data/eval_results/`)
   - those JSON files go into the **PDFs/XML/CSV/JSON input of the Ingestion pipeline** (the Part 1 landing
     folder), so they flow through Text Extraction → Chunking → Embeddings → Metadata Enrichment
     (`modality="eval"`, score, date) → Vector Database
   - because they are tagged `modality="eval"`, Metadata Filtering can include, weight or exclude them
6. Swap the direct LLM call from Part 2 for this pipeline behind the same function.

**Done when:** every answer passes through LLM (with MCP/TOOLS) → GUARDRAIL → EVAL → RESULT; unsafe answers come
back with a reason; safe results are cached; and the eval JSON is stored separately and shows up as retrievable
context.

---

## Implementation notes (these don't change the architecture)

- **Vector Database = the Elasticsearch index**, accessed through `langchain-elasticsearch`. The diagram feeds the
  Vector Database into Elasticsearch Hybrid Search, so one ES index with `dense_vector` fields plays both roles.
- **CLIP and text embeddings are different vector spaces.** Keep them in separate fields and fuse only their
  rankings (Rank Fusion), never the raw scores.
- **Eval results in the corpus.** Tag them (`modality="eval"`, score) so low-scoring ones can be filtered out and
  never outrank original documents by accident.

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
- **Ingestion:** 0 calls. Metadata Enrichment would add one call per chunk only if LLM-written summaries or
  keywords were added, and this plan doesn't add them.

Ways to keep the count down without changing the diagram:
- **GUARDRAIL:** use a small safety classifier (0 LLM calls).
- **Query Enhancement:** use a small, fast model, since it runs on every query.
- **EVAL:** run cheap checks inline (e.g. matching citations against chunk ids) and the full LLM judge in the
  background or on a sample.

With the classifier guardrail and background EVAL, a typical miss is **2 + N** calls in the request path
(Query Enhancement + LLM).

## Open decisions

- **Kafka.** The empty `Streaming/kafka_producers/` folder was removed in the restructure because Kafka is not in
  the diagram. To use Kafka, add it to the diagram first (e.g. as the stream source feeding the Spark Streaming
  boxes), then to this plan. Until then, Part 1.6 uses the file-based stream source (`data/landing/`) and no Kafka
  code is written.
