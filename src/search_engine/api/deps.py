from fastapi import Request

from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.retrieval.pipeline import SearchPipeline


def get_pipeline(request: Request) -> SearchPipeline:
    """The one SearchPipeline built at startup (models are loaded once and shared by all requests)."""
    return request.app.state.pipeline


def get_ingestion(request: Request) -> IngestionPipeline:
    """The one IngestionPipeline built at startup; it shares the models with the SearchPipeline."""
    return request.app.state.ingestion
