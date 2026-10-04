# The API image: FastAPI + the Ingestion, Retrieval and LLM pipelines. Built and started by docker-compose.yml.
#
# Two stages: the build stage installs the locked dependencies with uv; the run stage copies only the finished
# virtual environment and the source, so no build tools or caches end up in the image.

# ---- build ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (the slow part), from uv.lock only: a code change doesn't reinstall them.
# The uv cache is a build cache mount, so the downloads are never stored in the image.
# On Linux, pyproject.toml takes PyTorch from the CPU-only index (no CUDA, several GB smaller).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- run ----
FROM python:3.12-slim

# libgomp: OpenMP, used by PyTorch and faiss on CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv

# Models (bge, CLIP, cross encoder, Whisper) are downloaded on first use into HF_HOME, a volume in
# docker-compose.yml, so they are fetched once and kept, and the image stays small.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false
RUN mkdir -p /models /app/data && chown app:app /models /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "search_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
