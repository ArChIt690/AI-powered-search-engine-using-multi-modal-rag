"""FastAPI app: the USER entry point. Run with `uv run uvicorn search_engine.api.app:app`."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from search_engine.api.routes import ingest, search
from search_engine.core.logging import setup_logging
from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.retrieval.pipeline import SearchPipeline


def create_app(
    pipeline_factory: Callable[[], SearchPipeline] | None = None,
    ingestion_factory: Callable[[], IngestionPipeline] | None = None,
) -> FastAPI:
    """The factories build the two pipelines at startup (tests pass fake ones)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = (pipeline_factory or _warm_pipeline)()
        # bge and CLIP come from infra/models.py, so ingestion reuses the models the search pipeline loaded
        app.state.ingestion = (ingestion_factory or IngestionPipeline)()
        yield

    app = FastAPI(title="AI-powered multimodal search engine", lifespan=lifespan)
    app.include_router(search.router)
    app.include_router(ingest.router)
    return app


def _warm_pipeline() -> SearchPipeline:
    """Builds the pipeline and loads bge, CLIP and the cross encoder now, not on the first user's request."""
    setup_logging()
    pipeline = SearchPipeline()
    pipeline.reranker.search.text_store.embeddings.embed_query("warm up")
    pipeline.reranker.search.image_store.embeddings.embed_query("warm up")
    pipeline.reranker.cross_encoder.score([("warm up", "warm up")])  # the first call is several times slower
    return pipeline


app = create_app()
