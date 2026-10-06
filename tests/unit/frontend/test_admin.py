"""The admin page and the identity rules, run by Streamlit's test harness with a fake API client."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "frontend"))

import shared  # noqa: E402
from tests.unit.frontend.test_app import FakeClient, app, texts  # noqa: E402

ADMIN = {"login": "Archit@Example.com", "name": "Archit"}
USER = {"login": "someone@example.com", "name": "Someone"}


@pytest.mark.parametrize(
    ("user", "admins", "allowed"),
    [
        (None, "", True),  # opened on the server itself (127.0.0.1): no Tailscale identity
        (ADMIN, "archit@example.com, other@example.com", True),  # logins are compared case-insensitively
        (USER, "archit@example.com", False),
        (ADMIN, "", False),  # through Tailscale with no SEARCH_ADMINS set: nobody
    ],
)
def test_who_is_an_admin(monkeypatch, user, admins, allowed):
    monkeypatch.setenv("SEARCH_ADMINS", admins)

    assert shared.is_admin(user) is allowed


def test_a_non_admin_tailscale_user_is_turned_away(monkeypatch):
    monkeypatch.setattr(shared, "viewer", lambda: USER)
    monkeypatch.setenv("SEARCH_ADMINS", "archit@example.com")
    client = FakeClient()

    at = app(client, "admin.py")

    assert "someone@example.com is not an admin" in at.error[0].value
    assert len(at.file_uploader) == 0 and client.ingested == []


def test_an_admin_uploads_documents_and_the_list_refreshes(monkeypatch):
    monkeypatch.setattr(shared, "viewer", lambda: ADMIN)
    monkeypatch.setenv("SEARCH_ADMINS", "archit@example.com")
    client = FakeClient()
    at = app(client, "admin.py")
    assert at.title[0].value == "Admin: documents" and "Signed in as Archit" in texts(at.caption)
    assert next(b for b in at.button if b.label == "Ingest").disabled  # nothing chosen yet

    at.file_uploader[0].upload("new_notes.txt", b"Goroutines are lightweight.", "text/plain").run()
    next(b for b in at.button if b.label == "Ingest").click().run()

    assert client.ingested == [[("new_notes.txt", b"Goroutines are lightweight.")]]
    assert at.success[0].value == "Ingested 1 file(s) into 2 chunks."
    assert "Ingested documents (3)" in [heading.value for heading in at.subheader]


def test_the_admin_sees_how_to_fix_services_that_are_down():
    at = app(FakeClient(up=False), "admin.py")

    assert len(at.sidebar.children) == 0
    assert at.warning[0].value == "Not running: Search API, Elasticsearch, Redis. Start the services: `docker compose up -d`"
