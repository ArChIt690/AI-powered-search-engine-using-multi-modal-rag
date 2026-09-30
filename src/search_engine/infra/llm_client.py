"""The chat models, built once. Two free providers (Groq and Gemini) are used together.

Each job ("role") has a preferred provider and falls back to the other when a call fails (outage, rate limit,
unusable output), so neither free rate limit is reached quickly and one provider being down doesn't stop search:

    role     used by                          first choice          falls back to
    fast     QUERY Enhancement                Groq, fast model      Gemini
    guard    GUARDRAIL                        Groq, safety model    Gemini
    answer   the LLM (agent)                  Groq, main model      Gemini
    judge    EVAL                             Gemini                Groq, fast model

EVAL runs on the other provider than the answer, so an answer isn't judged by the model that wrote it.
A provider without an API key is left out; with one key, every role uses that provider.
"""

from functools import lru_cache
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from search_engine.core.config import get_settings
from search_engine.core.exceptions import LLMUnavailableError

Role = Literal["fast", "guard", "answer", "judge"]
Provider = Literal["groq", "gemini"]

_ORDER: dict[Role, tuple[Provider, ...]] = {
    "fast": ("groq", "gemini"),
    "guard": ("groq", "gemini"),
    "answer": ("groq", "gemini"),
    "judge": ("gemini", "groq"),
}


def chat_models(role: Role = "answer") -> list[BaseChatModel]:
    """The role's models, preferred provider first; only providers with an API key."""
    models = [model for provider in _ORDER[role] if (model := _model(provider, role))]
    if not models:
        raise LLMUnavailableError("No LLM API key is set: add GROQ_API_KEY or GEMINI_API_KEY to .env")
    return models


def get_chat_model(role: Role = "answer") -> Runnable:
    """The role's chat model, falling back to the other provider when a call fails."""
    return _with_fallbacks(chat_models(role))


def get_structured_model(schema: type, role: Role = "fast") -> Runnable:
    """As `get_chat_model`, but returning `schema` objects.

    json_schema output: the model fills the schema directly. With tool calling, gpt-oss (Groq) sometimes names the
    tool "functions.<Schema>" and LangChain rejects the reply.
    """
    return _with_fallbacks([model.with_structured_output(schema, method="json_schema") for model in chat_models(role)])


def _with_fallbacks(runnables: list[Any]) -> Runnable:
    first, *rest = runnables
    return first.with_fallbacks(rest) if rest else first


def _model(provider: Provider, role: Role) -> BaseChatModel | None:
    s = get_settings()
    if provider == "groq":
        names = {"answer": s.groq_model, "guard": s.groq_guard_model}
        return _build("groq", names.get(role, s.groq_fast_model)) if s.groq_api_key else None
    return _build("google_genai", s.gemini_model) if s.gemini_api_key else None


@lru_cache
def _build(langchain_provider: str, name: str) -> BaseChatModel:
    """One client per model, shared by every role that uses it."""
    s = get_settings()
    key = s.groq_api_key if langchain_provider == "groq" else s.gemini_api_key
    return init_chat_model(
        name,
        model_provider=langchain_provider,
        api_key=key.get_secret_value(),
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )
