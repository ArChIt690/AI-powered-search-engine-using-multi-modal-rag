"""The Elasticsearch client, built once and shared by the LangChain vector stores."""

from functools import lru_cache

from elasticsearch import Elasticsearch

from search_engine.core.config import get_settings


@lru_cache
def get_es_client() -> Elasticsearch:
    # Bulk writes of large files (thousands of chunks with vectors) can take longer than the 10 s default.
    return Elasticsearch(get_settings().es_url, request_timeout=60)
