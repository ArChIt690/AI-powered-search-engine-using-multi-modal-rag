"""What the search page (app.py) and the admin page (admin.py) share: who is signed in, a notice when the search
engine is down, and the list of ingested files."""

import os
from typing import Any

import streamlit as st

from api_client import ApiClient, ApiError

# Tailscale (`tailscale serve`) adds the signed-in tailnet user to every request it forwards to the page.
LOGIN_HEADER = "Tailscale-User-Login"
NAME_HEADER = "Tailscale-User-Name"


def viewer() -> dict[str, str] | None:
    """The Tailscale user looking at the page ({"login", "name"}), or None when it's opened without Tailscale
    (http://localhost on the server itself)."""
    headers = st.context.headers
    login = headers.get(LOGIN_HEADER)
    if not login:
        return None
    return {"login": login, "name": headers.get(NAME_HEADER) or login}


def admin_logins() -> set[str]:
    """SEARCH_ADMINS: comma-separated Tailscale logins allowed on the admin page."""
    return {login.strip().lower() for login in os.environ.get("SEARCH_ADMINS", "").split(",") if login.strip()}


def is_admin(user: dict[str, str] | None) -> bool:
    """Through Tailscale: only the logins in SEARCH_ADMINS (none set = nobody). Without Tailscale the page is only
    reachable on the server itself (127.0.0.1), so whoever is there is the admin."""
    if user is None:
        return True
    return user["login"].lower() in admin_logins()


def service_notice(client: ApiClient, *, admin: bool = False) -> dict[str, bool]:
    """Nothing when the search engine is up; otherwise a warning (users: try later; the admin: how to start it)."""
    health = client.health()
    if not all(health.values()):
        down = [label for service, label in (("api", "Search API"), ("elasticsearch", "Elasticsearch"),
                                             ("redis", "Redis")) if not health.get(service)]
        st.warning(f"Not running: {', '.join(down)}. Start the services: `docker compose up -d`" if admin else
                   "The search engine is not fully running. Please try again later.")
    return health


def load_files(client: ApiClient, api_up: bool) -> list[dict[str, Any]]:
    """The ingested files, fetched once per visit and again after an ingest or a refresh."""
    state = st.session_state
    if "files" not in state and api_up:
        try:
            state.files = client.files()
        except ApiError:
            return []
    return state.get("files", [])


def files_table(files: list[dict[str, Any]], title: str) -> None:
    st.subheader(f"{title} ({len(files)})")
    if files:
        st.dataframe(
            [{"File": f["file_name"], "Passages": f["passages"], "Pictures": f["pictures"]} for f in files],
            hide_index=True,
        )
    else:
        st.caption("Nothing ingested yet.")
    if st.button("Refresh list"):  # e.g. after an admin ingested from the command line
        st.session_state.pop("files", None)
        st.rerun()
