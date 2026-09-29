"""FastAPI app: the USER entry point. Run with `uv run uvicorn search_engine.api.app:app`."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from search_engine.api.routes import search
from search_engine.core.logging import setup_logging
from search_engine.retrieval.pipeline import SearchPipeline


def create_app(pipeline_factory: Callable[[], SearchPipeline] | None = None) -> FastAPI:
    """`pipeline_factory` builds the pipeline at startup (tests pass a fake one)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = (pipeline_factory or _warm_pipeline)()
        yield

    app = FastAPI(title="AI-powered multimodal search engine", lifespan=lifespan)
    app.include_router(search.router)
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
