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

1. Install Docker Desktop.
2. Copy `.env.example` to `.env` and add your free API keys: `GROQ_API_KEY` (console.groq.com) and
   `GEMINI_API_KEY` (aistudio.google.com). With only one key, that provider does everything.

## Run the project

```
docker compose up -d --build
```

Then open the admin page, http://localhost:8502, to add documents, and the search page, http://localhost:8501, to
ask questions. Both answer only on this machine; to share them, see "Share it over Tailscale" below.

- The first start takes a few minutes: the images are built and the API downloads its models (about 1.5 GB) into
  a Docker volume. Later starts take about a minute.
- `docker compose ps` shows whether all services are healthy; `docker compose logs -f api` shows the API.
- `docker compose down` stops everything. Your files, indexes, caches and models are kept (`./data` and Docker
  volumes) and are there again on the next `up`.
- If Docker Desktop has stopped, open it and run `docker compose up -d` again.

### Share it over Tailscale (HTTPS, private to your tailnet)

Users only ask questions; admins add documents. Tailscale publishes just the two pages, never the databases or
the API, and only to devices in your tailnet:

| Page | Address | Who |
|---|---|---|
| Search | `https://search.<your-tailnet>.ts.net` | everyone in the tailnet |
| Admin | `https://search.<your-tailnet>.ts.net:8443` | logins listed in `SEARCH_ADMINS` |

1. In the Tailscale admin console: **DNS → Enable HTTPS**, then **Settings → Keys → Generate auth key**.
2. In `.env`: `TS_AUTHKEY=tskey-auth-...` and `SEARCH_ADMINS=you@example.com` (no spaces around `=`).
3. `docker compose --profile tailscale up -d --build`. The `search` machine joins the tailnet; the first visit
   takes about half a minute while its certificate is issued.

The pages know who is visiting from Tailscale (`Tailscale-User-Login`), so the admin page lets in only
`SEARCH_ADMINS`. For a network-level lock as well, apply `tailscale/policy.example.hujson` (only admins can reach
port 8443) and set `TS_EXTRA_ARGS=--advertise-tags=tag:search-engine` with a tagged auth key.

### Without Docker for the app (for development)

Install [uv](https://docs.astral.sh/uv/) and run `uv sync` once. Then, in three terminals:

```
docker compose up -d elasticsearch redis           # only the databases
uv run uvicorn search_engine.api.app:app           # the API, on http://localhost:8000
uv run streamlit run frontend/app.py               # the search page, on http://localhost:8501
uv run streamlit run frontend/admin.py --server.port 8502   # the admin page
```

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
