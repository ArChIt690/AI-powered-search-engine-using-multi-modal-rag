# Build Order

This plan follows `Search_Engine_Architecture.drawio` exactly. Every step maps to a box or arrow in the diagram.
Components are first written as plain Python functions, and in Phase 7 they are wrapped in Spark Streaming
(every bold "Spark Streaming:" box in the diagram). Nothing in the architecture is dropped or merged.

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

---

## Phase 0: Setup ✅

- `docker-compose.yml`: Elasticsearch + Redis
- `Search_Engine/config.py`: pydantic-settings reading `.env`
- `uv add "elasticsearch>=8,<9" redis pydantic-settings`
- Check: `curl http://localhost:9200`, `docker exec search-redis redis-cli ping`

## Phase 1: Ingestion, text path

Diagram boxes: *Text Documents, PDFs/XML/CSV/JSON (text part), Text Extraction, Chunking (Smart, Semantic),
Text Embeddings, Metadata Enrichment, Vector Database.*

1. `Schema/chunk.py`: the `Chunk` model: `id`, `text`, `source`, `modality` (text / image / video_transcript / video_frame / eval),
   `page`, `timestamp`, `metadata`, `text_embedding`, `image_embedding`. Every stage passes this model along.
2. `Data/loaders/`: **Text Extraction**. One loader per input type:
   - Text documents (`.txt`, `.md`)
   - PDFs: text only for now (image/chart/table separation is Phase 3)
   - XML, CSV, JSON: "for others, just text extraction"
3. `Data/chunking.py`: **Chunking (Smart, Semantic)**. Both parts of the box:
   - *Smart*: structure-aware splitting (headings, paragraphs, table rows, CSV rows, JSON objects) with overlap
   - *Semantic*: split where embedding similarity between neighbouring sentences drops
4. `Data/embed.py`: **Text Embeddings** (sentence-transformers, e.g. `BAAI/bge-small-en-v1.5`).
5. `Data/enrichment.py`: **Metadata Enrichment**. Runs *after* embedding, as in the diagram: source, file type,
   modality, page, dates, title/section, and optionally LLM-generated keywords/summary. Metadata Filtering in
   Phase 2 depends on these fields.
6. `Data/index.py`: **Vector Database**. LangChain's `ElasticsearchStore` over one ES index: a `text` field (BM25,
   English analyzer), a `text_embedding` `dense_vector` and typed `metadata.*` fields. The diagram feeds this
   database into Elasticsearch Hybrid Search, so the ES index is the vector database. The `image_embedding`
   field is added to the mapping in Phase 3.

**Done when:** you can ingest a folder of txt/pdf/xml/csv/json files and see enriched, embedded chunks in ES.

## Phase 2: Retrieval pipeline, search path

Diagram boxes: *USER → QUERY, Query Enhancement, Metadata Filtering, Elasticsearch Hybrid Search
(Keyword BM25 + Semantic), Reranking (Rank Fusion + Cross Encoder), LLM.*

1. `Schema/query.py`, `Schema/response.py`: request/response models.
2. `Retrieval/query_enhance.py`: **Query Enhancement** (LLM rewrite, expansion, or HyDE).
3. `Retrieval/filters.py`: **Metadata Filtering**. Turns the query or user options into ES `filter` clauses on the
   fields from Metadata Enrichment.
4. `Retrieval/search.py`: **Elasticsearch Hybrid Search** through LangChain's `ElasticsearchStore`. Two sub-searches,
   as in the diagram:
   - **Keyword Search (BM25)** on `text` (`BM25Strategy`)
   - **Semantic Search**, kNN on `text_embedding` (`DenseVectorStrategy`)
   - Elasticsearch's built-in RRF needs a paid license (the local basic license returns 403), so the two lists
     are fused in the next step. LangChain's hybrid mode also returns no scores.
5. `Retrieval/rerank.py`: **Reranking**. First **Rank Fusion** (RRF, in our code) of the BM25 and semantic lists,
   then a **Cross Encoder** (e.g. `BAAI/bge-reranker-base`) over the fused top-N.
6. `Agent/llm.py`: a basic LLM call with the reranked chunks as context (the full LLM architecture comes in Phase 4).
7. `Search_Engine/api/app.py`: FastAPI `/search` endpoint, so USER → QUERY → answer works end to end.
8. `eval/`: 20–30 questions with known answers to measure recall@k and MRR while tuning this phase
   (a dev tool, separate from the EVAL box in Phase 4).

**Done when:** a question returns a cited answer produced from reranked hybrid results.

## Phase 3: Ingestion, multimodal paths

Diagram boxes: *PDF separation of images/charts/tables, PICTURES, Image Embeddings (CLIP), VIDEO,
Extract Audio & Convert to Text, Takes pictures frame by frame temporarily.*

1. `Data/loaders/pdf.py`: **"for PDFs separating images, charts, tables"**
   - text → Chunking (Phase 1 path)
   - tables → serialized to text (markdown) → Chunking
   - **charts, images from the PDFs → PICTURES** (the arrow in the diagram)
2. `Data/image_embed.py`: **PICTURES → Image Embeddings (CLIP)** → Metadata Enrichment → Vector Database
   (add an `image_embedding` `dense_vector` to the existing mapping with `put_mapping`, and write to it with a
   second `ElasticsearchStore` whose `vector_query_field="image_embedding"`).
3. `Data/loaders/video.py`: **VIDEO** splits two ways:
   - **Extract Audio & Convert to Text** (ffmpeg + Whisper) → the transcript joins **Text Extraction**, then the normal text path
   - **Takes pictures frame by frame, temporarily** (keyframe sampling to a temp dir) → **PICTURES** → CLIP;
     delete the temp frames after embedding
   - keep video timestamps in metadata for both branches
4. Retrieval update: **Semantic Search** also queries `image_embedding`, using the CLIP text encoder on the query,
   and those results go into Rank Fusion.

**Done when:** a text query can return a PDF chart, an image, or a video moment.

## Phase 4: LLM architecture

Diagram boxes: *LLM, MCP, TOOLS, GUARDRAIL (SAFE / UNSAFE), EVAL, RESULT, FINAL RESULT to USER.*

1. `Agent/llm.py`: the **LLM** as an agent that can call:
   - `Agent/mcp.py`: **MCP** client (connect to MCP servers)
   - `Agent/tools.py`: **TOOLS** (local function tools)
2. `Agent/guardrail.py`: **GUARDRAIL** on the LLM output:
   - **UNSAFE** → return to the USER with the reason ("sends the reason alongside")
   - **SAFE** → EVAL
3. `Agent/eval.py`: **EVAL** (LLM-as-judge: faithfulness to the context, relevance, citation correctness), output as JSON.
4. `Agent/pipeline.py`: builds the **RESULT**, which goes back as the **FINAL RESULT to USER**.

**Done when:** every answer passes through guardrail → eval → result, and unsafe answers come back with a reason.

## Phase 5: Caching

Diagram boxes: *FAISS Semantic Cache, Redis Prompt Caching, Sessional Queries, RESULT → "goes for caching".*
Caches sit **after Reranking and before the LLM**, in the order the diagram shows.

1. `Retrieval/semantic_cache.py`: **FAISS Semantic Cache**, "checks if a similar question is in the cache"
   - HIT → return to the USER
   - MISS → Redis
2. `Retrieval/prompt_cache.py`: **Redis Prompt Caching**, "checks if the exact prompt or a similar prompt is there"
   - exact: hash of the normalized prompt; similar: Redis vector search (redis-stack) over prompt embeddings
   - HIT → return to the USER
   - MISS → LLM
3. `Retrieval/session.py`: **Sessional Queries**. The USER's queries in the current session are stored in Redis
   (keyed by session id, with a TTL). They feed the prompt cache and give the LLM conversation context.
4. Write-back: **RESULT → FAISS Semantic Cache** ("goes for caching"). Only results that passed GUARDRAIL + EVAL
   are cached.

**Done when:** a repeated or similar question is answered from the cache without reaching the LLM.

## Phase 6: Eval feedback loop

Diagram: *"all the eval results are stored separately"* and *"the EVAL result goes in a JSON format to the RAG
pipeline for context, so that it's better"* (the arrow from EVAL back to the PDFs/XML/CSV/JSON input).

1. `Agent/eval.py`: save every eval result as a JSON record in its own store (e.g. an `eval_results` ES index or a
   `Data/eval_results/` folder), stored **separately**.
2. Feed those JSON records into the **JSON input of the Ingestion pipeline**, so they go through
   Text Extraction → Chunking → Embeddings → Metadata Enrichment (with `modality="eval"`, score and date) → Vector DB.
3. They then take part in retrieval as context. Because they are tagged `modality="eval"`, Metadata Filtering can
   include, weight or exclude them.

**Done when:** eval JSON is stored separately and shows up as retrievable context.

## Phase 7: Spark Streaming

Diagram boxes (all bold): *Spark Streaming: Text Extraction, Chunking, Text Embeddings, Image Embeddings (CLIP),
Extract Audio & Convert to Text.*

1. `Streaming/spark_jobs/`: one Spark Structured Streaming job per box above, each calling the Phase 1–3 functions
   (pure functions make this mostly wiring).
2. Stream source: a watched landing folder (Spark file source), so new files, including the eval JSON from
   Phase 6, are picked up continuously.
3. Add Spark to `docker-compose.yml` in this phase.

**Done when:** dropping a file into the landing folder makes it searchable without a manual ingest run.

---

## Implementation notes (these don't change the architecture)

- **Vector Database = the Elasticsearch index**, accessed through LangChain's `langchain-elasticsearch`. The diagram
  feeds the Vector Database into Elasticsearch Hybrid Search; one ES index with `dense_vector` fields plays both roles.
- **CLIP and text embeddings are different vector spaces.** Keep them in separate fields and fuse only their
  rankings (Rank Fusion), never the raw scores.
- **Eval results in the corpus.** Tag them (`modality="eval"`, score) so low-scoring ones can be filtered out and
  never outrank original documents by accident.
- **Kafka is not in the diagram.** Phase 7 uses a file-based stream source. Add Kafka only if you also add it to
  the diagram.
