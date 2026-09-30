# Streamlit frontend, connected to the whole project: plan
n> Approved plan (2026-09-30).

## Context
All three diagram parts are built and tested (Ingestion, Retrieval, LLM Architecture), but they can only be used
from the CLI or by calling the API by hand. The user wants a **Streamlit frontend in its own folder**, connected to
everything, and the whole project working together.

The diagram doesn't change: the frontend is the screen for the **USER** box. It only talks to the FastAPI API and
holds no search logic, like the CLI. The backend runs once, so the models are loaded once. That matters on this
machine, which has run out of memory before.

```
Browser ─► Streamlit (frontend/) ─HTTP─► FastAPI (api/) ─► Ingestion pipeline ─► Elasticsearch
                                                     └─► Retrieval pipeline ─► LLM Architecture ─► answer
```

## 1. Backend: two small additions so the UI can do everything
Today the API has `POST /search` and `GET /health`. The UI also needs to add files and see what is there.

- **`api/routes/ingest.py`** (new):
  - `POST /ingest`: upload one or more files.
    - Each is saved to `data/landing/` under its bare file name (`Path(name).name`, so an upload can't write
      outside that folder).
    - It is then ingested with the existing `IngestionPipeline.ingest_path`.
    - Returns files, chunks, skipped, failed.
    - Re-uploading a file with the same name replaces its old chunks (same path → same `doc_id`, already built).
  - `GET /files`: the ingested files with their passage and picture counts.
- **`ingestion/vector_store.py`**: move the Elasticsearch "count chunks per file" query out of
  `llm/tools/list_files.py` into one function, `file_counts(settings)`. The tool and `GET /files` both use it.
- **`api/app.py` + `api/deps.py`**: build one `IngestionPipeline` at startup next to the `SearchPipeline`. The
  models are already shared through `infra/models.py`, so nothing loads twice.
- **`schemas/response.py`**: `IngestResponse`, `FileInfo`.
- **`core/config.py`**: `landing_dir = "data/landing"`.
- **New dependency:** `python-multipart` (FastAPI file uploads).

## 2. Frontend: `frontend/` (new folder at the project root)
- **`frontend/api_client.py`**: a small `httpx` client for the API.
  - Methods: `search()`, `ingest()`, `files()`, `health()`.
  - The API address comes from `API_URL` (default `http://localhost:8000`).
  - Long timeouts (search 120 s, ingest 15 min), and clear errors.
- **`frontend/app.py`**: the Streamlit app, run with `uv run streamlit run frontend/app.py`.
  - **Chat in the middle.** `st.chat_input` + `st.chat_message`, one `session_id` per conversation, so
    follow-up questions work (Sessional Queries). Each answer shows:
    - the answer text with its `[n]` marks
    - a badge for where it came from: LLM, FAISS cache, Redis cache, blocked, no results
    - **Sources:** one expander per citation, with file, page or video time, content type and snippet
    - Eval scores (faithfulness / relevance / citations, passed or not), the tools the LLM used, and "searched
      as: …" when the query was rewritten
    - a red box with the reason when the Guardrail blocked the answer
    - a warning when the LLM is unavailable (the API's 503 message)
  - **Sidebar:**
    - status of the API, Elasticsearch and Redis (`/health`)
    - **Add files:** uploader + "Ingest" button → `POST /ingest`, with a result summary
    - the list of ingested files (`/files`)
    - filters (file type, modality, content, file name) and `top_k`
    - "New conversation"
- **New dependency:** `streamlit`.

## 3. Docs
- `README.md`: a "Run the project" section with the four commands: `docker compose up -d`, the API, the frontend,
  and optionally the CLI.
- `Prompts/Build_order.md`: a "Frontend" section (the USER box's screen).
- This plan saved to `Prompts/plans/4_frontend.md`.
- `.env.example`: `API_URL`.

## Reused
- `SearchRequest` / `SearchResponse` / `SearchFilters`
- `IngestionPipeline.ingest_path` and `IngestReport`
- `SUPPORTED_EXTENSIONS` (the uploader's allowed types)
- the list-files ES query, moved to `file_counts`
- `create_app(pipeline_factory)` test pattern
- `get_es_client`, `get_redis`

## Verification ("the whole project should work")
- **Unit tests** (no Docker):
  - `/ingest`: saves and ingests uploads with a fake pipeline, strips folder parts from names, reports skipped
    and failed
  - `/files`
  - `api_client` against a fake HTTP transport: success, 503, connection refused
  - the Streamlit app with Streamlit's own `AppTest` and a fake client: a question renders the answer, sources,
    eval and tools; a blocked answer shows the reason; filters and session id are sent; upload calls ingest
- **Full-system check, for real:**
  1. `docker compose up -d`.
  2. Start the API (`uvicorn`) and Streamlit (check `/_stcore/health`).
  3. Through the API the UI uses: upload real files (md, csv, a PDF, an image) → `/files` lists them.
  4. Drive the real Streamlit app (`AppTest`) against the live API, with real Groq and Gemini:
     - ask a question → cited answer with eval scores
     - ask it again → served from the cache
     - ask a follow-up → rewritten using the session
     - ask a calculator question → tools shown
  5. `GET /health` is green, and the eval results file has the records.
  6. The whole test suite still passes: fast tests, plus the three end-to-end tests one at a time.
- If Docker Desktop has stopped again, it is restarted first. It stopped several times from memory pressure.

## Not included (say if you want them)
- Deleting a file from the UI.
- Showing the actual picture for image citations. PDF charts and video frames aren't stored as files, only their
  embeddings.
