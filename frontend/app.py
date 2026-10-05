"""The USER's screen: ask questions about the documents and see where each answer came from.

Users only ask; adding documents is the admin's job (admin.py). Run with `uv run streamlit run frontend/app.py`
(the API must be running; see README). Everything goes through the API (`api_client.py`); this file only draws
the page.
"""

import time
import uuid
from pathlib import PurePosixPath
from typing import Any

import streamlit as st

from api_client import ApiClient, ApiError
from shared import files_table, load_files, status_panel, viewer

SOURCES = {
    "llm": "Answered by the LLM",
    "faiss_semantic_cache": "From the FAISS semantic cache (a similar question was answered before)",
    "redis_prompt_cache": "From the Redis prompt cache (this prompt was answered before)",
    "guardrail_blocked": "Blocked by the guardrail",
    "no_results": "No matching documents",
}
MODALITIES = {
    "text": "Text", "image": "Images", "video_transcript": "Video speech", "video_frame": "Video frames",
}
CONTENTS = {
    "text": "Text", "table": "Tables", "record": "Records (CSV / JSON / XML)", "image": "Images",
    "chart": "Charts", "frame": "Video frames",
}


def main() -> None:
    st.set_page_config(page_title="AI Search Engine", page_icon=":mag:", layout="wide")
    state = st.session_state
    state.setdefault("client", ApiClient())
    state.setdefault("session_id", uuid.uuid4().hex)
    state.setdefault("messages", [])

    options = sidebar(state.client)

    st.title("AI Search Engine")
    st.caption("Ask questions about the company's documents: text, PDFs, tables, images and videos. Answers cite their sources.")
    if not state.messages:
        st.info("Ask a question below. The documents you can search are listed in the sidebar.")
    for message in state.messages:
        show_message(message)

    if question := st.chat_input("Ask a question about the documents"):
        user_message = {"role": "user", "content": question}
        state.messages.append(user_message)
        show_message(user_message)
        reply = ask(state.client, question, state.session_id, options)
        state.messages.append(reply)
        show_message(reply)


def ask(client: ApiClient, question: str, session_id: str, options: dict[str, Any]) -> dict[str, Any]:
    """One question through the API; failures become a message in the chat, not a crash."""
    started = time.monotonic()
    with st.spinner("Searching the documents..."):
        try:
            response = client.search(question, session_id, options["filters"], options["top_k"])
        except ApiError as error:
            return {"role": "assistant", "error": str(error), "llm_down": error.status == 503}
    return {"role": "assistant", "response": response, "seconds": time.monotonic() - started}


def show_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "error" in message:
            if message.get("llm_down"):
                st.warning(f"The LLM is not available right now. {message['error']}")
            else:
                st.error(message["error"])
        else:
            show_response(message["response"], message.get("seconds"))


def show_response(response: dict[str, Any], seconds: float | None) -> None:
    if response.get("guardrail_reason"):
        st.error(f"This answer was blocked by the safety check: {response['guardrail_reason']}")
    else:
        st.markdown(response["answer"])

    details = [SOURCES.get(response["source"], response["source"])]
    if seconds is not None:
        details.append(f"{seconds:.1f} s")
    st.caption(" · ".join(details))
    if response.get("enhanced_query") and response["enhanced_query"] != response["query"]:
        st.caption(f"Searched as: {response['enhanced_query']}")
    if response.get("tools_used"):
        st.caption("Tools used: " + ", ".join(response["tools_used"]))
    if scores := response.get("eval"):
        verdict = "passed" if scores["passed"] else "not passed, so not cached"
        st.caption(
            f"Eval: faithfulness {scores['faithfulness']}/5 · relevance {scores['relevance']}/5 · "
            f"citations {scores['citation_correctness']}/5 · {verdict}"
        )
        if scores.get("notes"):
            st.caption(f"Eval notes: {scores['notes']}")

    citations = response.get("citations") or []
    if citations:
        st.markdown(f"**Sources ({len(citations)})**")
        for citation in citations:
            with st.expander(citation_label(citation)):
                st.text(citation["snippet"])
                if citation.get("source"):
                    st.caption(citation["source"])


def citation_label(citation: dict[str, Any]) -> str:
    """[1] report.pdf, page 3 (table)   /   [2] review.mp4, at 01:15 (text)"""
    where = ""
    if citation.get("page") is not None:
        where = f", page {citation['page']}"
    elif citation.get("timestamp") is not None:
        seconds = int(citation["timestamp"])
        where = f", at {seconds // 60:02d}:{seconds % 60:02d}"
    return f"[{citation['number']}] {citation['file_name']}{where} ({citation['content']})"


def sidebar(client: ApiClient) -> dict[str, Any]:
    """Who is signed in, status, the searchable documents and the search options. Returns {"filters", "top_k"}."""
    state = st.session_state
    with st.sidebar:
        if user := viewer():
            st.markdown(f":material/person: Signed in as **{user['name']}**")
        health = status_panel(client)
        files = load_files(client, health["api"])
        files_table(files, "Documents")

        st.subheader("Search options")
        names = [f["file_name"] for f in files]
        types = sorted({PurePosixPath(name).suffix.lstrip(".").lower() for name in names if "." in name})
        filters = {
            "file_name": st.multiselect("Only these files", names),
            "file_type": st.multiselect("Only these file types", types),
            "modality": st.multiselect("Only this kind of content", list(MODALITIES), format_func=MODALITIES.get),
            "content": st.multiselect("Only these parts", list(CONTENTS), format_func=CONTENTS.get),
        }
        top_k = st.slider("Passages given to the LLM", min_value=1, max_value=20, value=10)

        if st.button("New conversation"):
            state.messages = []
            state.session_id = uuid.uuid4().hex
            st.rerun()
    return {"filters": {name: values for name, values in filters.items() if values}, "top_k": top_k}


main()
