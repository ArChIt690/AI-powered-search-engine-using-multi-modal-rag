# AI-powered search engine (multimodal RAG)

Ask questions about your own files and get answers that cite their sources. It works on text and Markdown files,
PDFs (text, tables, charts, images), CSV / JSON / XML, images, and videos (speech and frames).

The design is in `docs/Search_Engine_Architecture.drawio`, and `Prompts/Build_order.md` maps every box of it to
the code.

## How it works

```
Browser ─► Streamlit (frontend/) ─► FastAPI (src/search_engine/api/)
                                       ├─► Ingestion: read ─► chunk ─► embed (bge, CLIP) ─► Elasticsearch
                                       └─► Retrieval: enhance query ─► filter ─► hybrid search ─► rerank
                                             ─► FAISS cache ─► Redis cache ─► LLM (tools, MCP)
                                             ─► Guardrail ─► Eval ─► answer with citations
```

## Set up (once)

1. Install [uv](https://docs.astral.sh/uv/) and Docker Desktop.
2. `uv sync`
3. Copy `.env.example` to `.env` and add your free API keys: `GROQ_API_KEY` (console.groq.com) and
   `GEMINI_API_KEY` (aistudio.google.com). With only one key, that provider does everything.

## Run the project

Use three terminals, in the project folder:

```
docker compose up -d                              # 1. Elasticsearch and Redis
uv run uvicorn search_engine.api.app:app          # 2. the API, on http://localhost:8000 (loads the models: ~1 minute)
uv run streamlit run frontend/app.py              # 3. the frontend, on http://localhost:8501
```

Then open http://localhost:8501, add files in the sidebar, and ask a question.

If the page says the API or the databases are not running, start them with the commands above. If Docker Desktop
has stopped, open it and run `docker compose up -d` again. Your ingested files stay.

## Without the frontend

```
uv run search-engine ingest path/to/file-or-folder
uv run search-engine search "What was the revenue of the East region?" --session s1 --file-type pdf
```

The API has `POST /search`, `POST /ingest`, `GET /files` and `GET /health`. The interactive docs are at
http://localhost:8000/docs.

## Tests

```
uv run pytest                                 # fast tests (the ones that need Docker are skipped if it is down)
uv run pytest -m slow tests/unit              # real models
uv run pytest -m "slow and integration"       # end to end: one test per part of the diagram
uv run python -m eval.run                     # retrieval benchmark: recall@k and MRR, no LLM calls
```

On a machine with 16 GB of RAM, run the three end-to-end tests one file at a time.
