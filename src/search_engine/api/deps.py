from fastapi import Request

from search_engine.retrieval.pipeline import SearchPipeline


def get_pipeline(request: Request) -> SearchPipeline:
    """The one SearchPipeline built at startup (models are loaded once and shared by all requests)."""
    return request.app.state.pipeline
