"""USER -> QUERY -> answer, over HTTP. The routes only call the Retrieval pipeline."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from search_engine.api.deps import get_pipeline
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.infra.elasticsearch import get_es_client
from search_engine.infra.redis import get_redis
from search_engine.retrieval.pipeline import SearchPipeline
from search_engine.schemas.query import SearchRequest
from search_engine.schemas.response import SearchResponse

router = APIRouter()


# A plain `def`: the pipeline is blocking (models, Elasticsearch, Redis, LLM), so FastAPI runs it in a thread.
@router.post("/search")
def search(request: SearchRequest, pipeline: Annotated[SearchPipeline, Depends(get_pipeline)]) -> SearchResponse:
    try:
        return pipeline.search(request)
    except LLMUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/health")
def health() -> dict[str, bool]:
    return {"elasticsearch": _ok(lambda: get_es_client().ping()), "redis": _ok(lambda: get_redis().ping())}


def _ok(check) -> bool:
    try:
        return bool(check())
    except Exception:
        return False
