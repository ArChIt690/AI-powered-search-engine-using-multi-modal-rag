"""The chat models, built once. Any LangChain provider works; the free ones set up here are Gemini and Groq."""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from search_engine.core.config import get_settings


@lru_cache
def get_chat_model(fast: bool = False) -> BaseChatModel:
    """The answering LLM, or with `fast=True` the one for Query Enhancement (`llm_fast_model`, else the same)."""
    s = get_settings()
    extra = {"api_key": s.llm_api_key.get_secret_value()} if s.llm_api_key else {}  # else the provider's own env var
    return init_chat_model(
        (s.llm_fast_model or s.llm_model) if fast else s.llm_model,
        model_provider=s.llm_provider,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        **extra,
    )
