"""The Streamlit page, run by Streamlit's own test harness with a fake API client (no server, no browser)."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).parents[3] / "frontend"
sys.path.insert(0, str(FRONTEND))

import shared  # noqa: E402
from api_client import ApiError  # noqa: E402

ANSWER = {
    "query": "total revenue?", "enhanced_query": "What is the total revenue of all regions?",
    "answer": "The total is **436** [1].", "source": "llm", "session_id": "s", "guardrail_reason": None,
    "citations": [
        {"number": 1, "chunk_id": "r-0", "file_name": "report.pdf", "source": "/data/landing/report.pdf",
         "modality": "text", "content": "table", "page": 1, "timestamp": None, "snippet": "|East|143|"},
        {"number": 2, "chunk_id": "v-0", "file_name": "review.mp4", "source": "/data/landing/review.mp4",
         "modality": "video_transcript", "content": "text", "page": None, "timestamp": 75.0, "snippet": "Revenue grew."},
    ],
    "eval": {"faithfulness": 5, "relevance": 4, "citation_correctness": 5, "notes": "Slightly terse.", "passed": True},
    "tools_used": ["calculator"],
}


class FakeClient:
    def __init__(self, response=None, error=None, up=True):
        self.response, self.error, self.up = response or ANSWER, error, up
        self.searches, self.ingested = [], []
        self.file_list = [{"file_name": "report.pdf", "passages": 3, "pictures": 1},
                          {"file_name": "notes.md", "passages": 5, "pictures": 0}]

    def health(self):
        return {"api": self.up, "elasticsearch": self.up, "redis": self.up}

    def files(self):
        return list(self.file_list)

    def search(self, query, session_id=None, filters=None, top_k=None):
        self.searches.append({"query": query, "session_id": session_id, "filters": filters, "top_k": top_k})
        if self.error:
            raise self.error
        return self.response

    def ingest(self, files):
        self.ingested.append(files)
        self.file_list.append({"file_name": files[0][0], "passages": 2, "pictures": 0})
        return {"files": 1, "chunks": 2, "skipped": ["movie.exe"], "failed": {}} if len(files) > 1 else {
            "files": 1, "chunks": 2, "skipped": [], "failed": {}}


def app(client: FakeClient, page: str = "app.py") -> AppTest:
    at = AppTest.from_file(str(FRONTEND / page), default_timeout=30)
    at.session_state["client"] = client
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(elements) -> str:
    return "\n".join(str(element.value) for element in elements)


def test_page_has_no_sidebar_and_shows_a_hint_before_the_first_question():
    at = app(FakeClient())

    assert at.title[0].value == "AI Search Engine"
    assert len(at.sidebar.children) == 0
    assert at.info[0].value == "Ask a question below." and len(at.warning) == 0  # nothing is down
    assert not [b for b in at.button if b.label == "New conversation"]  # nothing to clear yet


def test_users_cannot_upload_documents():
    at = app(FakeClient())

    assert len(at.file_uploader) == 0
    assert not [b for b in at.button if b.label == "Ingest"]


def test_the_tailscale_user_is_shown(monkeypatch):
    monkeypatch.setattr(shared, "viewer", lambda: {"login": "asha@example.com", "name": "Asha Rao"})

    at = app(FakeClient())

    assert "Signed in as **Asha Rao**." in texts(at.caption)


def test_when_the_engine_is_down_users_get_a_plain_notice():
    at = app(FakeClient(up=False))

    assert at.warning[0].value == "The search engine is not fully running. Please try again later."


def test_a_question_shows_the_answer_its_sources_eval_and_tools():
    client = FakeClient()
    at = app(client)

    at.chat_input[0].set_value("total revenue?").run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "The total is **436** [1]." in texts(at.chat_message[1].markdown)
    captions = texts(at.chat_message[1].caption)
    assert "Answered by the LLM" in captions and "Searched as: What is the total revenue of all regions?" in captions
    assert "Tools used: calculator" in captions
    assert "Eval: faithfulness 5/5 · relevance 4/5 · citations 5/5 · passed" in captions and "Slightly terse." in captions
    labels = [expander.label for expander in at.chat_message[1].expander]
    assert labels == ["[1] report.pdf, page 1 (table)", "[2] review.mp4, at 01:15 (text)"]
    assert client.searches[0]["query"] == "total revenue?"


def test_a_conversation_keeps_one_session_and_shows_every_turn():
    client = FakeClient()
    at = app(client)

    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("and the second?").run()

    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    first, second = (search["session_id"] for search in client.searches)
    assert first == second and len(first) == 32  # follow-ups share the session (Sessional Queries)


def test_new_conversation_clears_the_chat_and_starts_another_session():
    client = FakeClient()
    at = app(client)
    at.chat_input[0].set_value("first").run()

    next(b for b in at.button if b.label == "New conversation").click().run()
    at.chat_input[0].set_value("again").run()

    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert client.searches[0]["session_id"] != client.searches[1]["session_id"]


def test_questions_search_all_documents_with_the_default_settings():
    client = FakeClient()
    at = app(client)

    at.chat_input[0].set_value("q").run()

    assert client.searches[0]["filters"] is None and client.searches[0]["top_k"] is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [("faiss_semantic_cache", "FAISS semantic cache"), ("redis_prompt_cache", "Redis prompt cache"),
     ("no_results", "No matching documents")],
)
def test_where_the_answer_came_from_is_shown(source, expected):
    at = app(FakeClient({**ANSWER, "source": source, "eval": None, "tools_used": []}))

    at.chat_input[0].set_value("q").run()

    assert expected in texts(at.chat_message[1].caption)


def test_a_blocked_answer_shows_the_reason_instead_of_the_answer():
    blocked = {**ANSWER, "source": "guardrail_blocked", "answer": "The answer was blocked by the safety check: ...",
               "guardrail_reason": "The answer contained an API key, which can't be shown.", "citations": [],
               "eval": None, "tools_used": []}
    at = app(FakeClient(blocked))

    at.chat_input[0].set_value("what is the key?").run()

    assert "blocked by the safety check: The answer contained an API key" in at.chat_message[1].error[0].value
    assert len(at.chat_message[1].expander) == 0 and "Blocked by the guardrail" in texts(at.chat_message[1].caption)


def test_llm_outage_and_other_errors_are_shown_in_the_chat():
    outage = app(FakeClient(error=ApiError("The LLM could not answer: 429 quota", status=503)))
    outage.chat_input[0].set_value("q").run()
    assert "The LLM is not available right now. The LLM could not answer: 429 quota" in outage.chat_message[1].warning[0].value

    down = app(FakeClient(error=ApiError("Can't reach the search engine API at http://localhost:8000.")))
    down.chat_input[0].set_value("q").run()
    assert "Can't reach the search engine API" in down.chat_message[1].error[0].value
    assert not down.exception
