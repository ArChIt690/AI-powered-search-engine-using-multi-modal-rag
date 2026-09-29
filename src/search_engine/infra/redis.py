"""The Redis client, built once: the index version, and the client the caches and sessions share."""

from functools import lru_cache

from redis import Redis

from search_engine.core.config import Settings, get_settings


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def index_version(settings: Settings | None = None) -> int:
    """The corpus version Ingestion bumps after every write; cached answers from an older version are stale."""
    settings = settings or get_settings()
    return int(get_redis().get(settings.index_version_key) or 0)
