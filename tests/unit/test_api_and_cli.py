import pytest
from fastapi.testclient import TestClient

from search_engine import cli
from search_engine.api import app as app_module
from search_engine.api.routes import search as search_route
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.schemas.query import SearchFilters, SearchRequest
from search_engine.schemas.response import AnswerSource, Citation, SearchResponse


def response(request: SearchRequest) -> SearchResponse:
    citation = Citation(number=1, chunk_id="d-1", file_name="report.pdf", source="/report.pdf", modality="text",
                        content="table", page=3, snippet="|North|120|")
    return SearchResponse(query=request.query, enhanced_query=request.query + "?", answer="North [1].",
                          citations=[citation], source=AnswerSource.LLM, session_id=request.session_id)


class FakePipeline:
    def __init__(self, error=None):
        self.error, self.requests = error, []

    def search(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return response(request)


def client(pipeline):
    return TestClient(app_module.create_app(lambda: pipeline))


def test_post_search_returns_the_cited_answer():
    pipeline = FakePipeline()
    with client(pipeline) as c:
        reply = c.post("/search", json={"query": "who leads?", "session_id": "s1", "filters": {"file_type": ["pdf"]}})

    assert reply.status_code == 200
    body = reply.json()
    assert body["answer"] == "North [1]." and body["source"] == "llm" and body["citations"][0]["page"] == 3
    assert pipeline.requests[0].filters == SearchFilters(file_type=["pdf"])


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "q", "filters": {"author": ["me"]}}, {"query": "q", "top_k": 0}])
def test_bad_requests_are_rejected(body):
    with client(FakePipeline()) as c:
        assert c.post("/search", json=body).status_code == 422


def test_llm_outage_is_a_503_with_the_reason():
    with client(FakePipeline(LLMUnavailableError("The LLM could not answer: 429"))) as c:
        reply = c.post("/search", json={"query": "q"})

    assert reply.status_code == 503 and "429" in reply.json()["detail"]


def test_health_reports_each_service(monkeypatch):
    class Down:
        def ping(self):
            raise ConnectionError

    class Up:
        def ping(self):
            return True

    monkeypatch.setattr(search_route, "get_es_client", lambda: Up())
    monkeypatch.setattr(search_route, "get_redis", lambda: Down())
    with client(FakePipeline()) as c:
        assert c.get("/health").json() == {"elasticsearch": True, "redis": False}


def test_cli_options_become_the_search_request():
    parser_args = [
        "search", "which region leads?", "--session", "s1", "--file-type", "pdf", "--file-type", "csv",
        "--modality", "text", "--content", "table", "--top-k", "5",
    ]
    captured = {}

    class Capture:
        def search(self, request):
            captured["request"] = request
            return response(request)

    import search_engine.retrieval.pipeline as pipeline_module

    original = pipeline_module.SearchPipeline
    pipeline_module.SearchPipeline = Capture
    try:
        cli.main(parser_args)
    finally:
        pipeline_module.SearchPipeline = original

    request = captured["request"]
    assert request.query == "which region leads?" and request.session_id == "s1" and request.top_k == 5
    assert request.filters == SearchFilters(file_type=["pdf", "csv"], modality=["text"], content=["table"])


def test_cli_prints_answer_citations_and_source():
    text = cli.format_response(response(SearchRequest(query="who leads")))

    assert "North [1]." in text and "(searched as: who leads?)" in text
    assert "[1] report.pdf, page 3 (table): '|North|120|'" in text and "answered by: llm" in text


def test_cli_exits_with_the_reason_when_the_llm_is_down(monkeypatch):
    import search_engine.retrieval.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "SearchPipeline", lambda: FakePipeline(LLMUnavailableError("429 quota")))

    with pytest.raises(SystemExit, match="429 quota"):
        cli.main(["search", "q"])
