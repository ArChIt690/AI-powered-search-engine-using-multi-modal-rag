import pytest

from search_engine.core.config import Settings
from search_engine.infra import llm_client


@pytest.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "init_chat_model", lambda model, **kwargs: calls.append((model, kwargs)) or object())
    llm_client.get_chat_model.cache_clear()
    yield calls
    llm_client.get_chat_model.cache_clear()


def test_builds_the_configured_provider_and_model_with_the_key(monkeypatch, calls):
    settings = Settings(_env_file=None, llm_provider="groq", llm_model="openai/gpt-oss-120b", llm_api_key="k")
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    llm_client.get_chat_model()

    model, kwargs = calls[0]
    assert model == "openai/gpt-oss-120b"
    assert kwargs["model_provider"] == "groq" and kwargs["api_key"] == "k"
    assert kwargs["temperature"] == 0.0 and kwargs["timeout"] == 30.0 and kwargs["max_retries"] == 2


def test_fast_model_is_separate_and_defaults_to_the_main_model(monkeypatch, calls):
    monkeypatch.setattr(
        llm_client, "get_settings", lambda: Settings(_env_file=None, llm_model="main", llm_fast_model=None)
    )
    llm_client.get_chat_model(fast=True)
    monkeypatch.setattr(
        llm_client, "get_settings", lambda: Settings(_env_file=None, llm_model="main", llm_fast_model="small")
    )
    llm_client.get_chat_model.cache_clear()
    llm_client.get_chat_model(fast=True)

    assert [model for model, _ in calls] == ["main", "small"]


def test_defaults_are_groq_with_a_small_model_for_query_enhancement(monkeypatch, calls):
    monkeypatch.setattr(llm_client, "get_settings", lambda: Settings(_env_file=None))

    llm_client.get_chat_model()
    llm_client.get_chat_model(fast=True)

    assert [(model, kwargs["model_provider"]) for model, kwargs in calls] == [
        ("openai/gpt-oss-120b", "groq"), ("openai/gpt-oss-20b", "groq")
    ]


def test_no_key_leaves_it_to_the_providers_env_var(monkeypatch, calls):
    monkeypatch.setattr(llm_client, "get_settings", lambda: Settings(_env_file=None, llm_api_key=None))

    llm_client.get_chat_model()

    assert "api_key" not in calls[0][1]


def test_model_is_built_once(monkeypatch, calls):
    monkeypatch.setattr(llm_client, "get_settings", lambda: Settings(_env_file=None))

    assert llm_client.get_chat_model() is llm_client.get_chat_model()
    assert len(calls) == 1
