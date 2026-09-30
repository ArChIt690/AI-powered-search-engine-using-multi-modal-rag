"""The frontend's only link to the backend: a small HTTP client for the FastAPI API.

The frontend imports nothing from `search_engine`, so it loads no models and could run on another machine; set
`API_URL` to where the API is (default http://localhost:8000).
"""

import os
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"
_SEARCH_TIMEOUT_S = 120  # a new question makes several LLM calls
_INGEST_TIMEOUT_S = 900  # a long PDF or a video takes minutes to chunk, embed and transcribe


class ApiError(Exception):
    """The API could not do what was asked; `str(error)` is a message fit to show the user."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str | None = None, *, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, transport=transport)

    def health(self) -> dict[str, bool]:
        """{"api", "elasticsearch", "redis"}: what is up. Never raises, so the page can always show a status."""
        try:
            services = self._request("GET", "/health", timeout=5)
        except ApiError:
            return {"api": False, "elasticsearch": False, "redis": False}
        return {"api": True, **services}

    def files(self) -> list[dict[str, Any]]:
        """The ingested files: [{"file_name", "passages", "pictures"}]."""
        return self._request("GET", "/files", timeout=15)

    def search(
        self, query: str, session_id: str | None = None, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, Any]:
        """USER -> QUERY: the API's SearchResponse (answer, citations, source, eval, tools_used, ...)."""
        body = {"query": query, "session_id": session_id, "filters": filters or {}, "top_k": top_k}
        return self._request("POST", "/search", json=body, timeout=_SEARCH_TIMEOUT_S)

    def ingest(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Uploads (file name, content) pairs: {"files", "chunks", "skipped", "failed"}."""
        uploads = [("files", (name, content)) for name, content in files]
        return self._request("POST", "/ingest", files=uploads, timeout=_INGEST_TIMEOUT_S)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise ApiError("The search engine took too long to answer. Try again.") from error
        except httpx.HTTPError as error:
            raise ApiError(
                f"Can't reach the search engine API at {self.base_url}. Is it running? "
                "(uv run uvicorn search_engine.api.app:app)"
            ) from error
        if response.is_success:
            return response.json()
        raise ApiError(_detail(response), response.status_code)


def _detail(response: httpx.Response) -> str:
    """FastAPI's error message: a string for our errors, a list of field problems for a bad request."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"The API answered with an error ({response.status_code})."
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or f"The API answered with an error ({response.status_code}).")
