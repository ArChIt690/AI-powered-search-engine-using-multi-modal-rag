# Build Order

This plan follows **`AI_Search_Engine_Architecture_v2-1.drawio`** ("Architecture v2") strictly. Every step maps
to a box or arrow in that diagram. Nothing in the diagram is dropped, merged or reordered, and nothing that is not
in the diagram is added. The v1 diagram (`docs/Search_Engine_Architecture.drawio`) is superseded and no longer
used for decisions.

## Rules

1. **Each region of the diagram is built end to end.** The regions are: Ingestion pipeline, Request path,
   Cache layer, Retrieval pipeline, Generation (LLM architecture), Frontend, Backend infrastructure,
   Observability stack and Offline eval pipeline. A part is done only when every box and arrow in its region
   exists, is connected to its neighbouring regions, and runs (from the CLI, the API or the UI). Nothing is left
   as a stub "for a later phase".
2. **The diagram decides.** If the code or this file disagrees with the diagram, the diagram wins. To change the
   architecture, change the diagram first, then this file, then the code.
3. **Every bold "Spark Streaming:" box is a Spark job.** Each box's logic is written as a plain Python function
   (so it can be tested without Spark), and the PySpark Structured Streaming job in `streaming/jobs/` calls that
   function. Both are built in the same part.
4. **Languages follow the diagram.** The API GATEWAY is **Go (Gin)**. The Frontend is **Next.js**. Everything
   else (ingestion, caches, retrieval, generation, evals) is **Python**. The LLM is **API-only**; embeddings,
   CLIP, Whisper and the cross-encoder run on our own server (v2 note 6).

## Target flows (from the diagram)

**Ingestion pipeline** (unchanged from v1)
```
Text Documents ──► Spark: Text Extraction ──► Spark: Chunking (Smart, Semantic) ──► Spark: Text Embeddings ──┐
PDFs/XML/CSV/JSON ─► PDFs: separate images, charts, tables │ others: just text extraction ─► Chunking          │
                         └─ charts, images from the PDFs ─► PICTURES                                          ├─► Metadata Enrichment ─► Vector Database
PICTURES ─────────────────────────────────────────► Spark: Image Embeddings (CLIP) ──────────────────────────┘
VIDEO ─┬─► Spark: Extract Audio & Convert to Text ─► Text Extraction
       └─► Takes pictures frame by frame temporarily ─► PICTURES
FRONTEND ── file upload ──► Ingestion
```

**Request path** (frontend → gateway)
```
USER ── query ──► FRONTEND (Next.js) ── HTTPS · reply via SSE ──► API GATEWAY (Go, Gin: auth · validate · route)
  ─► SESSION CHECK (session_id · conversation_id; Redis hot / Postgres history)
  ─► RATE LIMIT & QUOTAS (Redis token bucket, per-user daily token budget) ──over quota──► 429 Too Many Requests → user
  ─allowed─► INPUT GUARDRAIL (blocks before any retrieval) ──UNSAFE──► refuse + reason → user
  ─SAFE─► QUERY ENHANCEMENT (rewrites follow-ups into a standalone query using history)
  ─standalone query─► Cache layer
```

**Cache layer** (checked BEFORE retrieval)
```
REDIS EXACT CACHE (first, O(1); key = hash(query + filters + user scope); invalidated on re-ingest via TTL / index version)
   ──HIT──► stream cached answer → FRONTEND
   ──MISS─► FAISS SEMANTIC CACHE (second; similarity ≥ threshold; same key scoping as Redis)
               ──HIT──► FRONTEND
               ──MISS─► Retrieval pipeline
```

**Retrieval pipeline** (unchanged boxes)
```
METADATA FILTERING ─► ELASTICSEARCH HYBRID SEARCH (BM25 keyword + semantic dense, over the Vector DB)
  ─► RERANKING (rank fusion + cross-encoder) ── top-k chunks ──► Generation
Vector Database ── indexed chunks + embeddings ──► Elasticsearch Hybrid Search
```

**Generation** (LLM architecture)
```
CONTEXT BUILDING (system prompt + last N turns / summary + top-k chunks + question → one prompt)
  ─► LLM GATEWAY (LiteLLM: API keys · retries · provider fallback on 429/outage · token + cost logging)
  ─► MODEL ROUTER ──easy──► FAST API MODEL (cheap / free-tier, low latency)
                  └─hard──► LARGE API MODEL (2nd provider = fallback)          [MODEL POOL, API only]
MODEL POOL ◄── tool calls ──► MCP ◄──► TOOLS (call API · external search · run code in a Docker sandbox)
MODEL POOL ─► OUTPUT GUARDRAIL (safety + policy) ──UNSAFE──► refuse / regenerate
                                                 └─SAFE──► RESULT (stream to user · save turn → Postgres)
RESULT ── FINAL RESULT → user (SSE stream) ──► FRONTEND
RESULT ┄┄ async ┄┄► write Redis + FAISS (the Cache layer)
RESULT ┄┄ async ┄┄► ONLINE EVAL (light groundedness check, never blocks the reply)
                        ┄┄ eval results + 👍/👎 ┄┄► Postgres   (NOT back into ingestion)
```

**Backend infrastructure**: REDIS (sessions · rate limits · exact-match cache), FAISS (semantic-cache index),
ELASTICSEARCH / VECTOR DB (chunks · embeddings · metadata), POSTGRESQL (users · conversations · settings ·
eval_results · feedback).

**Observability stack**: LANGFUSE (traces · token usage · cost per request), PROMETHEUS + GRAFANA (latency ·
errors · provider rate-limit hits), STRUCTURED LOGS (JSON with request_id). Every stage emits OpenTelemetry spans.

**Offline eval pipeline**: BATCH EVAL (RAGAS-style): fixed test set + sampled traces + 👍/👎 → scores in Postgres →
dashboard; tracks quality release to release.

## Folder layout

```
gateway/                 Go module: the API GATEWAY (Gin), incl. Session Check and Rate Limit & Quotas middleware
frontend/                Next.js chat UI (the FRONTEND box)
src/search_engine/       Python package, one folder per diagram region
  core/                  config.py (pydantic-settings reading .env), logging.py, exceptions.py
  schemas/               Pydantic models shared by all regions (Document, Chunk, query, response, eval)
  infra/                 clients built in one place: Elasticsearch, Redis, FAISS, Postgres, models.py (bge, CLIP, Whisper, cross-encoder)
  ingestion/             Ingestion pipeline (sources/, chunking, text_embed, image_embed, enrichment, vector_store, pipeline)
  streaming/             spark.py (SparkSession builder) and jobs/: one PySpark job per bold "Spark Streaming:" box
  request/               Python half of the Request path: input_guardrail.py, query_enhance.py
  cache/                 Cache layer: exact_cache.py (Redis), semantic_cache.py (FAISS), keys.py (key scoping, index version)
  retrieval/             Retrieval pipeline: filters.py, hybrid_search.py, rerank.py, pipeline.py
  generation/            Generation: context.py, llm_gateway.py (LiteLLM), router.py, mcp.py, tools/, output_guardrail.py, result.py, online_eval.py, pipeline.py
  api/                   internal FastAPI service the gateway routes to (not exposed publicly)
  cli.py                 search-engine command: ingest, stream, search
db/migrations/           Postgres schema (users, conversations, turns, settings, eval_results, feedback)
eval/                    Offline eval pipeline (RAGAS-style batch eval)
observability/           Prometheus scrape config, Grafana dashboards, OpenTelemetry collector config
tests/unit/              mirrors src/ and gateway/, no Docker needed
tests/integration/       real Elasticsearch, Redis and Postgres from docker-compose.yml
tests/e2e/               query in → streamed answer out; one test per part's "Done when"
data/                    git-ignored runtime data: landing/ (Spark stream source), checkpoints/
docker/                  Dockerfile.api, Dockerfile.spark, Dockerfile.gateway, Dockerfile.frontend
```

The empty placeholder files left over from the v1 layout (`retrieval/prompt_cache.py`, `retrieval/session.py`,
`retrieval/semantic_cache.py`, `llm/*`, `schemas/eval.py`) are moved or deleted when the part that owns them is
built. Entry points (`gateway/`, `api/`, `cli.py`, `streaming/jobs/`) only call a region's `pipeline.py`; they
hold no logic. Run Python with `uv run search-engine ...` / `uv run pytest`, and Go with `go run ./cmd/gateway` /
`go test ./...` inside `gateway/`.

---

## Part 0: Setup ✅

- `docker-compose.yml`: Elasticsearch + Redis. Postgres, Spark, the gateway, the frontend and the observability
  services are added by the part that first needs them.
- `core/config.py`: pydantic-settings reading `.env` (keys listed in `.env.example`)
- Check: `curl http://localhost:9200`, `docker exec search-redis redis-cli ping`

---

## Part 1: Ingestion pipeline (end to end)

Diagram boxes: *Text Documents, PDFs/XML/CSV/JSON, PICTURES, VIDEO, Spark Streaming: Text Extraction,
PDF separation of images/charts/tables, Spark Streaming: Extract Audio & Convert to Text, Takes pictures frame by
frame temporarily, Spark Streaming: Chunking (Smart, Semantic), Spark Streaming: Text Embeddings,
Spark Streaming: Image Embeddings (CLIP), Metadata Enrichment, Vector Database; arrow "file upload → ingestion".*

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
- *Semantic* for prose (txt, Markdown, PDF text): cut where the meaning shifts; oversized pieces re-split
- *Smart* for records (whole records packed, oversized records split at field lines with their first line
  repeated), tables (split between rows, header repeated) and transcripts (kept one-to-one with timestamps)
- chunks never cross a page, heading, transcript section or table edge

### 1.3 Text Embeddings and Image Embeddings (CLIP)
- `ingestion/text_embed.py`: **Text Embeddings** (sentence-transformers, `BAAI/bge-small-en-v1.5`, 384-dim) →
  `Chunk.text_embedding`; also the LangChain `Embeddings` used by semantic chunking and query embedding
- `ingestion/image_embed.py`: **Image Embeddings (CLIP)** (sentence-transformers `clip-ViT-B-32`, 512-dim):
  each `Picture` → a `Chunk` with `image_embedding`; plus the CLIP text encoder for text-to-image search
- models loaded once in `infra/models.py`

### 1.4 Metadata Enrichment
`ingestion/enrichment.py`, after both embedding boxes: chunk id, source, file type, modality, page, timestamp,
dates, title/section, and the **index version** the Cache layer uses for invalidation. Metadata Filtering in
Part 2 depends on these fields.

### 1.5 Vector Database
`ingestion/vector_store.py`: one Elasticsearch index with a `text` field (BM25, English analyzer),
`text_embedding` and `image_embedding` `dense_vector` fields, and typed `metadata.*` fields. Each successful
ingest bumps the index version (so the Cache layer's "invalidate on re-ingest" works).

### 1.6 Pipeline and Spark Streaming
- `ingestion/pipeline.py`: `IngestionPipeline` connects 1.1 → 1.5 for one file or a folder
- `streaming/jobs/`: one PySpark Structured Streaming job per bold box (**Text Extraction, Extract Audio & Convert
  to Text, Chunking, Text Embeddings, Image Embeddings (CLIP)**), each calling the functions above
- Stream source: the watched landing folder `data/landing/` (Spark file source, checkpoints in `data/checkpoints/`)
- Add Spark to `docker-compose.yml` (`docker/Dockerfile.spark`, Java 17)
- `search-engine ingest <path>` for a one-off run, `search-engine stream` to start the Spark jobs

**Done when:** dropping a txt/md/pdf/xml/csv/json file, an image or a video into `data/landing/` (or running
`search-engine ingest`) produces enriched, embedded chunks in Elasticsearch: text chunks, PDF tables, PDF charts
and images, video transcript chunks and video frames, each with the right `modality`. (The "file upload →
ingestion" arrow is wired in Part 5, when the frontend and gateway exist: uploads land in `data/landing/`.)

---

## Part 2: Retrieval pipeline (end to end)

Diagram boxes: *Metadata Filtering, Elasticsearch Hybrid Search (BM25 keyword + semantic dense), Reranking (rank
fusion + cross-encoder); arrows "indexed chunks + embeddings" (Vector DB → hybrid search), "top-k chunks" →
Context Building.*

1. `schemas/query.py`: the retrieval request (standalone query, filters, user scope, top_k).
2. `retrieval/filters.py`: **Metadata Filtering**: query options → ES `filter` clauses on the Part 1.4 fields
   (modality, file type, source, dates…).
3. `retrieval/hybrid_search.py`: **Elasticsearch Hybrid Search** over the Vector Database:
   - **BM25 keyword** on `text`
   - **semantic (dense)**: kNN on `text_embedding` (bge query embedding), plus kNN on `image_embedding` (CLIP
     text encoder) so text queries find PDF charts, images and video frames
   - ES's built-in RRF needs a paid license, so the lists are fused in the next step
4. `retrieval/rerank.py`: **Reranking**: rank fusion (RRF, our code) of the BM25, text-kNN and image-kNN lists,
   then a **cross-encoder** (`BAAI/bge-reranker-base`, local) over the fused top-N → **top-k chunks**.
5. `retrieval/pipeline.py`: filters → hybrid search → reranking; `search-engine search "<query>" --retrieve-only`
   prints the top-k chunks with citations.

**Done when:** a query returns reranked top-k chunks mixing text and image hits, each with its citation (source,
PDF page, video timestamp), and metadata filters narrow the results.

---

## Part 3: Generation (end to end)

Diagram boxes: *Context Building, LLM Gateway (LiteLLM), Model Router, Model Pool (Fast API model, Large API
model), MCP, TOOLS, Output Guardrail (UNSAFE → refuse / regenerate), RESULT (stream to user, save turn →
Postgres), Online Eval (async) → Postgres; arrows "top-k chunks", "tool calls", "FINAL RESULT → user (SSE
stream)", "write Redis + FAISS" (async), "eval results + 👍/👎" → Postgres.*

Infrastructure added: **PostgreSQL** in `docker-compose.yml`, schema in `db/migrations/` (users, conversations,
turns, settings, eval_results, feedback), client in `infra/postgres.py`.

1. `generation/context.py`: **Context Building**: system prompt + last N turns (or a summary) from Postgres +
   top-k chunks + question → one prompt with numbered citations.
2. `generation/llm_gateway.py`: **LLM Gateway (LiteLLM)**: provider API keys, retries, provider fallback on
   429 / outage, token + cost logging per call.
3. `generation/router.py`: **Model Router**: classifies the query easy / hard → **Fast API model** (cheap or
   free-tier) or **Large API model** (its 2nd provider is the fallback). Both are API-only.
4. `generation/mcp.py` and `generation/tools/`: **MCP** client and **TOOLS** (call API, external search, run code
   in a Docker sandbox), invoked through the model's tool calls.
5. `generation/output_guardrail.py`: **Output Guardrail** (safety + policy): **UNSAFE → refuse / regenerate**,
   **SAFE → RESULT**.
6. `generation/result.py`: **RESULT**: streams the answer token by token (SSE), saves the turn to Postgres, and
   asynchronously writes it to the Cache layer (Redis + FAISS; wired in Part 4).
7. `generation/online_eval.py`: **Online Eval (async)**: a light groundedness check that never blocks the reply;
   results go to Postgres `eval_results`, **never back into ingestion** (v2 note 4).
8. `generation/pipeline.py` + `api/routes/`: the internal FastAPI service streams RESULT over SSE;
   `search-engine search "<query>"` runs retrieval + generation from the CLI.

**Done when:** a question returns a streamed, cited answer from the routed model (with tool calls when needed);
unsafe output is refused or regenerated; the turn is saved in Postgres; and an online-eval row appears in
Postgres without delaying the answer.

---

## Part 4: Cache layer (end to end)

Diagram boxes: *Redis Exact Cache (checked first), FAISS Semantic Cache (checked second); arrows "standalone
query" in, HIT → stream cached answer to the frontend, MISS → next box / Metadata Filtering, "write Redis +
FAISS" from RESULT.*

1. `cache/keys.py`: cache key scoping = hash(standalone query + filters + user scope), plus the current index
   version from Part 1.5.
2. `cache/exact_cache.py`: **Redis Exact Cache**: O(1) lookup by key; entries expire by TTL and are invalidated
   when the index version changes (re-ingest). HIT → stream the cached answer; MISS → FAISS.
3. `cache/semantic_cache.py`: **FAISS Semantic Cache**: nearest cached query (bge embedding) with
   similarity ≥ threshold, same key scoping as Redis. HIT → cached answer; MISS → Metadata Filtering (Part 2).
   The FAISS index is persisted to a volume (`infra/faiss_index.py`).
4. Wire RESULT's async "write Redis + FAISS" (Part 3) and put the Cache layer in front of retrieval in the
   pipeline.

**Done when:** asking the same question again is answered from Redis, a paraphrase is answered from FAISS, both
without search, cross-encoder or LLM calls; and re-ingesting a file invalidates the cached answers.

---

## Part 5: Request path + API Gateway (Go) (end to end)

Diagram boxes: *USER, FRONTEND → API GATEWAY (HTTPS, reply via SSE), Session Check, Rate Limit & Quotas (429 →
user), Input Guardrail (UNSAFE → refuse + reason → user), Query Enhancement → "standalone query" into the Cache
layer; arrow "file upload → ingestion".*

1. **`gateway/`: API GATEWAY in Go (Gin)**, the only public entry point:
   - **auth**: JWT bearer tokens; users in Postgres
   - **validate**: request body schema and size limits, upload type and size limits
   - **route**: reverse-proxies to the internal FastAPI service, passing the SSE stream through unbuffered;
     file uploads are forwarded to the ingestion upload endpoint, which saves them into `data/landing/`
   - adds a `request_id` to every request and forwards it downstream
   - `docker/Dockerfile.gateway`, service in `docker-compose.yml`
2. **Session Check** (Go middleware, `gateway/internal/session`): resolves `session_id` and `conversation_id`,
   hot state in Redis, history in Postgres.
3. **Rate Limit & Quotas** (Go middleware, `gateway/internal/ratelimit`): Redis token bucket per user plus a
   per-user daily token budget (decremented from the token counts the LLM Gateway logs). Over quota →
   **429 Too Many Requests → user**.
4. `request/input_guardrail.py`: **Input Guardrail**, first Python step, before any retrieval or cache lookup:
   **UNSAFE → refuse + reason → user**; SAFE → Query Enhancement.
5. `request/query_enhance.py`: **Query Enhancement**: rewrites follow-ups into a standalone query using the
   conversation history → the Cache layer (Part 4).

**Done when:** an authenticated HTTPS request through the Go gateway streams an answer back over SSE; a request
without a valid token is rejected; exceeding the rate limit returns 429; an unsafe query is refused with a reason
before any retrieval; a follow-up question is rewritten into a standalone query; and a file uploaded through the
gateway gets ingested.

---

## Part 6: Frontend (end to end)

Diagram box: *FRONTEND (Next.js chat UI)* with its listed features.

- `frontend/`: Next.js chat UI: answers streamed token by token (SSE from the gateway), conversation sidebar,
  citations that open the PDF page / image / video timestamp, file upload (→ ingestion), 👍/👎 feedback
  (→ Postgres `feedback`), badges showing cached · model · latency.
- `docker/Dockerfile.frontend`, service in `docker-compose.yml`.

**Done when:** a user can log in, ask questions in a conversation, watch answers stream with clickable citations,
see cached/model/latency badges, upload a file and later find it in answers, and leave 👍/👎 feedback.

---

## Part 7: Observability stack (end to end)

Diagram boxes: *Langfuse, Prometheus + Grafana, Structured Logs; "every stage emits OpenTelemetry spans".*

- **OpenTelemetry spans** from every stage: Go gateway (otelgin), request path, cache, retrieval, generation,
  ingestion.
- **Langfuse**: traces, token usage and cost per request (fed by the LLM Gateway).
- **Prometheus + Grafana**: latency, errors, provider rate-limit hits; dashboards in `observability/`.
- **Structured logs**: JSON logs with `request_id` in Go (`log/slog`) and Python (`core/logging.py`).
- Services added to `docker-compose.yml`.

**Done when:** one request can be followed end to end by its `request_id` across gateway and Python logs, shows
as a trace with token cost in Langfuse, and moves the latency/error/rate-limit panels in Grafana.

---

## Part 8: Offline eval pipeline (end to end)

Diagram box: *BATCH EVAL (RAGAS-style): fixed test set + sampled traces + 👍/👎 → scores in Postgres → dashboard;
tracks quality release to release.*

- `eval/`: fixed test set (questions with known answers and sources), plus sampled production traces and 👍/👎
  feedback from Postgres.
- RAGAS-style metrics (faithfulness, answer relevance, context precision / recall), scores written to Postgres
  tagged with the release, and a Grafana dashboard comparing releases.

**Done when:** one command runs the batch eval, stores scores per release in Postgres, and the dashboard shows
the change from the previous release.

---

## Implementation notes (these don't change the architecture)

- **Vector Database = the Elasticsearch index.** One index with `dense_vector` fields serves both the Vector
  Database box and Elasticsearch Hybrid Search.
- **CLIP and text embeddings are different vector spaces.** Keep them in separate fields and fuse only their
  rankings (rank fusion), never the raw scores.
- **The internal FastAPI service** is how the Go gateway reaches the Python boxes; it is not public and holds no
  logic of its own.
- **Eval results never enter the corpus** (v2 note 4). `Modality.EVAL` from the v1 design is removed when
  Metadata Enrichment (1.4) is built.
- **Redis** no longer needs vector search (v2 uses Redis for exact-match only; FAISS does similarity), so plain
  Redis is enough; the current redis-stack image also works.

## LLM call budget

LLM calls per query in the v2 flow. Embeddings, CLIP, Whisper and the cross-encoder are local models, not LLM calls.

| Box | LLM calls | Notes |
|---|---|---|
| Input Guardrail | 1 (or 0) | 0 with a small local safety classifier. |
| Query Enhancement | 0–1 | Only follow-ups need rewriting; a first message can pass through unchanged. |
| Redis Exact / FAISS Semantic Cache | 0 | Hash lookup / embedding lookup. |
| Metadata Filtering, Hybrid Search, Reranking | 0 | |
| Model Router | 0 (or 1) | 0 with rules or a small classifier. |
| Model Pool | 1 + N | N = MCP / TOOLS round-trips. |
| Output Guardrail | 1 (or 0) | 0 with a local classifier; "regenerate" adds another Model Pool call. |
| Online Eval | 0–1, async | Never in the request path. |

- **Cache hit:** 0–2 calls (guardrail + enhancement). Because the caches sit before retrieval, a hit also skips
  search and the cross-encoder.
- **Cache miss:** 1 + N calls with classifier guardrails and a first message; up to 4 + N with LLM-based
  guardrails, enhancement and routing.
- **Ingestion:** 0 calls.

## Open decisions

- **Kafka.** Not in the diagram. Ingestion streams from the `data/landing/` folder (Spark file source). To use
  Kafka, add it to the diagram first.
- **Semantic chunker.** `SemanticChunker` comes from `langchain-experimental`, which is being retired. Decide in
  1.3 whether to replace it with our own splitter on the real bge model.
