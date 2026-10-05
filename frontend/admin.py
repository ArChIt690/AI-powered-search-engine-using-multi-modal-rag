"""The admin's screen: add documents to the search engine and see what is ingested.

Two locks keep users out: over Tailscale, the access policy lets only the admin group reach this page's port
(8443), and the page itself also checks the signed-in Tailscale login against SEARCH_ADMINS. Run with
`uv run streamlit run frontend/admin.py` (the API must be running).
"""

import streamlit as st

from api_client import ApiClient, ApiError
from shared import files_table, is_admin, load_files, status_panel, viewer


def main() -> None:
    st.set_page_config(page_title="AI Search Engine · Admin", page_icon=":material/admin_panel_settings:")
    state = st.session_state
    state.setdefault("client", ApiClient())
    state.setdefault("upload_round", 0)

    user = viewer()
    if not is_admin(user):
        st.error(f"{user['login']} is not an admin of this search engine.")
        st.caption("Ask an admin to add you to SEARCH_ADMINS. To ask questions, use the search page.")
        st.stop()

    st.title("Admin: documents")
    st.caption(f"Signed in as {user['name']}" if user else "Opened on the server itself (no Tailscale sign-in).")

    with st.sidebar:
        health = status_panel(state.client, admin=True)

    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Text, Markdown, PDF, CSV, JSON, XML, images or videos. A file with the same name replaces the old one.",
        accept_multiple_files=True,
        key=f"uploads_{state.upload_round}",  # a new key clears the uploader after an ingest
    )
    if st.button("Ingest", disabled=not uploads, type="primary"):
        ingest(state.client, uploads)
    if notice := state.pop("ingest_notice", None):
        kind, text = notice
        getattr(st, kind)(text)
    st.caption(
        "Large folders can also be ingested on the server: copy them into `data/landing/` and run "
        "`docker compose exec api search-engine ingest /app/data/landing`."
    )

    files_table(load_files(state.client, health["api"]), "Ingested documents")


def ingest(client: ApiClient, uploads) -> None:
    state = st.session_state
    with st.spinner(f"Ingesting {len(uploads)} file(s)... large PDFs and videos take a few minutes"):
        try:
            report = client.ingest([(upload.name, upload.getvalue()) for upload in uploads])
        except ApiError as error:
            state.ingest_notice = ("error", f"Ingest failed: {error}")
            return
    problems = [f"{name}: unsupported file type" for name in report["skipped"]]
    problems += [f"{name}: {reason}" for name, reason in report["failed"].items()]
    summary = f"Ingested {report['files']} file(s) into {report['chunks']} chunks."
    state.ingest_notice = ("warning", summary + " Not ingested: " + "; ".join(problems)) if problems else ("success", summary)
    state.upload_round += 1
    state.pop("files", None)  # reload the list
    st.rerun()


main()
