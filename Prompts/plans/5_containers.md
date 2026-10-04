# Containerize the whole project: plan
n> Approved plan (2026-10-04).

## Context
Today only Elasticsearch and Redis run in Docker; the API and the Streamlit frontend are started by hand with `uv`.
The goal: **one command, `docker compose up -d --build`, starts the whole project**, and it keeps working after a
restart with nothing lost.

Two lessons from earlier on this project shape the plan:
- The old Spark image grew to **10.3 GB** because uv's download cache was baked into it.
- Docker Desktop has **crashed several times from memory pressure** on this 16 GB machine.

## What runs (docker-compose.yml)
```
browser ─► frontend (Streamlit, :8501) ─► api (FastAPI, :8000) ─┬─► elasticsearch (:9200)
                                                                 ├─► redis (:6379)
                                                                 └─► Groq / Gemini (internet)
```
- `elasticsearch`, `redis`: unchanged.
- **`api` (new):** the FastAPI app with all three pipelines.
  - Starts after Elasticsearch and Redis report healthy.
  - Its own health check calls `/health`.
- **`frontend` (new):** the Streamlit page, with `API_URL=http://api:8000`. Starts after the API is healthy.
- Inside Docker, services find each other by name. The compose file sets `ES_URL=http://elasticsearch:9200` and
  `REDIS_URL=redis://redis:6379/0` for the API. No code changes: these are already settings.

## 1. API image: `Dockerfile` (project root)
Two stages, so the final image holds only what runs:
1. **Build stage** (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`):
   - `uv sync --frozen --no-dev`. The exact versions come from `uv.lock`, and there are no test tools.
   - uv's cache is a **build cache mount**, so it is never stored in the image (the 10 GB lesson).
   - The dependencies are installed before the source is copied, so a code change rebuilds in seconds.
   - **CPU-only PyTorch**, already set up in `pyproject.toml` for Linux (`[tool.uv.sources]`). That avoids several GB
     of CUDA.
2. **Run stage** (`python:3.12-slim`):
   - Copies the ready `.venv` and `src/`.
   - Runs `uvicorn search_engine.api.app:app --host 0.0.0.0 --port 8000` as a non-root user.

Expected size: about 2–2.5 GB (PyTorch CPU, sentence-transformers, faster-whisper, PyMuPDF, LangChain). I'll
report the real number.

## 2. Frontend image: `frontend/Dockerfile`
- `python:3.12-slim` + **only** `streamlit` and `httpx`, at the versions in `uv.lock`.
- The frontend imports nothing from the backend, so this image is small (about 0.4 GB).
- Runs `streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`.

## 3. What is kept between restarts (volumes)
| What | Where | Why |
|---|---|---|
| Elasticsearch data, Redis data | `es_data`, `redis_data` (as now) | your indexed files and caches |
| Uploaded files, FAISS cache, eval results | `./data` → `/app/data` | visible on your disk too, like now |
| Models (bge, CLIP, cross encoder, Whisper) | `model_cache` volume (Hugging Face cache) | downloaded **once** on first start, not baked into the image (keeps it ~1 GB smaller, rebuilds fast) |

## 4. Secrets and settings
- `.env` stays out of the image (already in `.dockerignore`) and is given to the API container through
  `env_file: .env`. Your Groq and Gemini keys are never inside an image.
- `.dockerignore`: also leave out `Prompts/`, `frontend/` (for the API image) and `eval/`, so the build context stays
  small.

## 5. Memory
Elasticsearch is capped at 1 GB already. The API needs about 1.5–2 GB (models + PyTorch). I'll add a memory limit
per service so one container can't take Docker Desktop down. If Docker's VM is still too small, I'll tell you the
one setting to raise (`.wslconfig` memory) rather than change it myself.

## 6. Docs
- `README.md` "Run the project" becomes `docker compose up -d --build` → http://localhost:8501, with the old `uv`
  commands kept as the "run without Docker" way.
- `Build_order.md`: a short "Containers" section.
- The plan is saved to `Prompts/plans/5_containers.md`.

## Verification
1. Start Docker Desktop (it is stopped right now) and check free disk space before building.
2. `docker compose build`, then report both image sizes.
3. `docker compose up -d`, and all four services report healthy.
4. Through the containers:
   - upload real files to `/ingest`
   - `/files` lists them
   - a question returns a cited answer with eval scores, from Groq and Gemini
   - the same question again comes from the cache
   - the Streamlit page answers at http://localhost:8501 (its health check, plus driving the page against the
     containerized API)
5. `docker compose restart`: the files, caches and models are still there, and no model is downloaded again.
6. The test suite still passes from the host. Development without containers still works with
   `docker compose up -d elasticsearch redis`.

## What changed while building (2026-10-04)
- **Redis lost its data on restart:** redis-stack's default snapshots run only every 5-60 minutes and its shutdown
  save didn't happen, so the index version, caches and sessions were gone after `docker compose down`/`up`.
  Fixed with the append-only file (`REDIS_ARGS=--appendonly yes --appendfsync everysec`): checked that a key and
  a cached answer survive a full restart.
- **Streamlit moved to the dev dependency group:** it pulled pandas and pyarrow into the API image (3.31 → 2.88 GB).
- **First start took ~8 min** (model downloads), so the API's health-check grace period is 15 min.
- **Project name pinned** in the compose file.
- Moving Docker's disk to E: is done in Docker Desktop (Settings → Resources → Advanced → Disk image location),
  not by hand: hand-moving the 27.6 GB `docker_data.vhdx` can leave Docker unable to start.
