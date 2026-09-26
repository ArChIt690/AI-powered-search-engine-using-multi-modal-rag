"""The Redis client, built once: index version now; sessions, rate limits and the exact cache later."""

from functools import lru_cache

from redis import Redis

from search_engine.core.config import get_settings


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)
