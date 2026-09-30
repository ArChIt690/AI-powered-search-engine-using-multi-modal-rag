import pytest
from langchain_core.runnables import RunnableLambda

from search_engine.core.config import Settings
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.infra import llm_client


class FakeModel:
    """Stands in for a provider's chat model: records what it was built with, and can be made to fail."""

    def __init__(self, name, kwargs, fail=False):
        self.name, self.kwargs, self.fail = name, kwargs, fail

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda _: self._reply(f"{schema.__name__} via {method} from {self.name}"))

    def with_fallbacks(self, fallbacks):
        return RunnableLambda(lambda x: self._reply(self.name)).with_fallbacks(
            [RunnableLambda(lambda x, m=m: m._reply(m.name)) for m in fallbacks]
        )

    def _reply(self, text):
        if self.fail:
            raise RuntimeError(f"429 from {self.name}")
        return text


@pytest.fixture
def build(monkeypatch):
    """Configures the client with `settings`; providers named in `failing` raise on every call."""

    def _build(failing=(), **settings):
        values = {"groq_api_key": "g", "gemini_api_key": "m", **settings}
        monkeypatch.setattr(llm_client, "get_settings", lambda: Settings(_env_file=None, **values))
        monkeypatch.setattr(
            llm_client, "init_chat_model",
            lambda name, **kwargs: FakeModel(name, kwargs, fail=kwargs["model_provider"] in failing),
        )
        llm_client._build.cache_clear()

    yield _build
    llm_client._build.cache_clear()


def names(role):
    return [(model.name, model.kwargs["model_provider"]) for model in llm_client.chat_models(role)]


def test_each_role_prefers_a_provider_and_falls_back_to_the_other(build):
    build()

    assert names("fast") == [("openai/gpt-oss-20b", "groq"), ("gemini-2.5-flash", "google_genai")]
    assert names("guard") == [("openai/gpt-oss-safeguard-20b", "groq"), ("gemini-2.5-flash", "google_genai")]
    assert names("answer") == [("openai/gpt-oss-120b", "groq"), ("gemini-2.5-flash", "google_genai")]
    assert names("judge") == [("gemini-2.5-flash", "google_genai"), ("openai/gpt-oss-20b", "groq")]


def test_models_get_their_own_key_and_the_shared_settings(build):
    build(groq_model="big", llm_timeout_s=12.0)

    groq, gemini = llm_client.chat_models("answer")

    assert groq.name == "big" and groq.kwargs["api_key"] == "g" and gemini.kwargs["api_key"] == "m"
    assert groq.kwargs["timeout"] == 12.0 and groq.kwargs["temperature"] == 0.0 and groq.kwargs["max_retries"] == 1


def test_a_provider_without_a_key_is_left_out(build):
    build(gemini_api_key=None)

    assert names("judge") == [("openai/gpt-oss-20b", "groq")]
    assert llm_client.get_chat_model("answer").name == "openai/gpt-oss-120b"  # no fallback wrapper needed


def test_no_keys_at_all_is_a_clear_error(build):
    build(groq_api_key=None, gemini_api_key=None)

    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY or GEMINI_API_KEY"):
        llm_client.get_chat_model("answer")


def test_a_failing_provider_falls_back_to_the_other(build):
    build(failing={"groq"})

    assert llm_client.get_chat_model("answer").invoke("q") == "gemini-2.5-flash"


def test_structured_output_uses_json_schema_and_falls_back_too(build):
    class Verdict:
        pass

    build()
    assert llm_client.get_structured_model(Verdict, "judge").invoke("q") == "Verdict via json_schema from gemini-2.5-flash"

    build(failing={"google_genai"})
    assert llm_client.get_structured_model(Verdict, "judge").invoke("q") == "Verdict via json_schema from openai/gpt-oss-20b"


def test_each_model_is_built_once(build):
    build()

    assert llm_client.chat_models("answer")[0] is llm_client.chat_models("answer")[0]
    assert llm_client.chat_models("fast")[0] is llm_client.chat_models("judge")[1]  # the same Groq fast model
