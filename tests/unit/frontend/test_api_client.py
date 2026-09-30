import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "frontend"))  # the frontend is a folder of scripts, not a package

from api_client import DEFAULT_API_URL, ApiClient, ApiError  # noqa: E402


def client(handler, base_url=None) -> ApiClient:
    return ApiClient(base_url, transport=httpx.MockTransport(handler))


def test_search_posts_the_question_session_filters_and_top_k():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={"answer": "North [1].", "source": "llm", "citations": []})

    reply = client(handler).search("who leads?", "s1", {"file_type": ["pdf"]}, 5)

    assert reply["answer"] == "North [1]."
    assert seen == {
        "path": "/search",
        "body": {"query": "who leads?", "session_id": "s1", "filters": {"file_type": ["pdf"]}, "top_k": 5},
    }


def test_ingest_uploads_every_file_as_multipart():
    seen = {}

    def handler(request):
        seen["type"], seen["body"] = request.headers["content-type"], request.content
        return httpx.Response(200, json={"files": 2, "chunks": 7, "skipped": [], "failed": {}})

    reply = client(handler).ingest([("notes.md", b"# Go"), ("people.csv", b"name,city")])

    assert reply["chunks"] == 7 and seen["type"].startswith("multipart/form-data")
    assert b'filename="notes.md"' in seen["body"] and b'filename="people.csv"' in seen["body"]
    assert seen["body"].count(b'name="files"') == 2


def test_files_and_health():
    def handler(request):
        if request.url.path == "/files":
            return httpx.Response(200, json=[{"file_name": "a.pdf", "passages": 3, "pictures": 1}])
        return httpx.Response(200, json={"elasticsearch": True, "redis": False})

    api = client(handler)

    assert api.files() == [{"file_name": "a.pdf", "passages": 3, "pictures": 1}]
    assert api.health() == {"api": True, "elasticsearch": True, "redis": False}


def test_an_unreachable_api_is_a_clear_error_and_health_never_raises():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    api = client(handler)

    assert api.health() == {"api": False, "elasticsearch": False, "redis": False}
    with pytest.raises(ApiError, match="Can't reach the search engine API at http://localhost:8000") as error:
        api.search("q")
    assert error.value.status is None


def test_api_errors_carry_the_apis_own_message_and_status():
    replies = {
        "/search": httpx.Response(503, json={"detail": "The LLM could not answer: 429 quota"}),
        "/files": httpx.Response(422, json={"detail": [{"msg": "String should have at least 1 character"}]}),
        "/ingest": httpx.Response(500, text="Internal Server Error"),
    }
    api = client(lambda request: replies[request.url.path])

    with pytest.raises(ApiError, match="429 quota") as llm_down:
        api.search("q")
    assert llm_down.value.status == 503
    with pytest.raises(ApiError, match="at least 1 character"):
        api.files()
    with pytest.raises(ApiError, match=r"error \(500\)"):
        api.ingest([("a.txt", b"x")])


def test_a_slow_api_is_a_timeout_message():
    def handler(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ApiError, match="took too long"):
        client(handler).search("q")


def test_api_url_comes_from_the_argument_then_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://search.internal:9000/")

    assert ApiClient().base_url == "http://search.internal:9000"
    assert ApiClient("http://other:1").base_url == "http://other:1"
    monkeypatch.delenv("API_URL")
    assert ApiClient().base_url == DEFAULT_API_URL
