from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.llm.context import AnswerContext
from search_engine.llm.tools import build_tools, calculator, list_files, read_surrounding
from search_engine.schemas.query import SearchFilters


def doc(chunk_id, text="text", **metadata):
    base = {"chunk_id": chunk_id, "file_name": "guide.md", "content": "text", "modality": "text"}
    return Document(text, metadata={**base, **metadata})


def context(chunks=(), **kwargs):
    return AnswerContext(Settings(es_text_index="t", es_image_index="i"), list(chunks), **kwargs)


def tool(name, ctx):
    return next(t for t in build_tools(ctx) if t.name == name)


def test_every_tool_has_a_name_and_instructions_for_the_llm():
    tools = build_tools(context())

    assert [t.name for t in tools] == ["search_documents", "read_surrounding", "list_files", "calculator"]
    assert all(len(t.description) > 60 for t in tools)


def test_added_passages_are_numbered_after_the_context_and_duplicates_are_skipped():
    ctx = context([doc("a"), doc("b")])

    shown = ctx.add([doc("b"), doc("c", "third"), doc("d", "fourth")])

    assert shown == "[3] (guide.md, text)\nthird\n\n[4] (guide.md, text)\nfourth"
    assert [d.metadata["chunk_id"] for d in ctx.chunks] == ["a", "b", "c", "d"]
    assert ctx.add([doc("a")]).startswith("No new passages")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("143 - 78", 65), ("(120 + 95) / 2", 107.5), ("143 / 436 * 100", 32.7981651376), ("2 ^ 10", 1024),
     ("1,200 * 3", 3600), ("-5 + 2", -3), ("7 // 2", 3), ("7 % 2", 1)],
)
def test_calculator_does_arithmetic(expression, expected):
    assert calculator.evaluate(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('dir')", "open('x')", "a + 1", "9 ** 999", "1 / 0", "2 +"])
def test_calculator_only_does_arithmetic_and_reports_what_it_cannot_do(expression):
    reply = tool("calculator", context()).invoke({"expression": expression})

    assert reply.startswith("Could not calculate")


def test_search_documents_runs_the_search_again_with_the_questions_filters():
    calls = []

    class Reranker:
        def retrieve(self, query, filters, top_k):
            calls.append((query, filters, top_k))
            return [doc("new", "found it")]

    filters = SearchFilters(file_type=["pdf"])
    ctx = context([doc("a")], filters=filters, reranker=Reranker())

    reply = tool("search_documents", ctx).invoke({"query": "south revenue"})

    assert calls == [("south revenue", filters, ctx.settings.agent_tool_top_k)]
    assert reply == "[2] (guide.md, text)\nfound it"


def test_read_surrounding_fetches_the_neighbouring_chunks_by_id(monkeypatch):
    asked = {}

    def mget(index, ids):
        asked.update(index=index, ids=ids)
        return {"docs": [
            {"_id": "d-3", "found": True, "_source": {"text": "before", "metadata": {"file_name": "guide.md", "content": "text"}}},
            {"_id": "d-5", "found": False},
        ]}

    monkeypatch.setattr(read_surrounding, "get_es_client", lambda: SimpleNamespace(mget=mget))
    ctx = context([doc("d-4", doc_id="d", chunk_index=4)])
    read = tool("read_surrounding", ctx)

    assert read.invoke({"passage_number": 1}) == "[2] (guide.md, text)\nbefore"
    assert asked == {"index": "t", "ids": ["d-3", "d-5"]}
    assert "no passage [7]" in read.invoke({"passage_number": 7})


def test_read_surrounding_of_the_first_chunk_only_asks_for_the_next(monkeypatch):
    asked = []
    monkeypatch.setattr(
        read_surrounding, "get_es_client",
        lambda: SimpleNamespace(mget=lambda index, ids: asked.extend(ids) or {"docs": [{"_id": ids[0], "found": False}]}),
    )

    reply = tool("read_surrounding", context([doc("d-0", doc_id="d", chunk_index=0)])).invoke({"passage_number": 1})

    assert asked == ["d-1"] and reply == "This passage has no neighbouring text."


def test_list_files_counts_passages_and_pictures_per_file(monkeypatch):
    buckets = {
        "t": [{"key": "report.pdf", "doc_count": 12}, {"key": "notes.md", "doc_count": 3}],
        "i": [{"key": "report.pdf", "doc_count": 2}],
    }
    search = lambda index, **kwargs: {"aggregations": {"files": {"buckets": buckets[index]}}}  # noqa: E731
    monkeypatch.setattr(list_files, "get_es_client", lambda: SimpleNamespace(search=search))

    reply = tool("list_files", context()).invoke({})

    assert reply == "2 files:\n- notes.md: 3 text passages\n- report.pdf: 12 text passages, 2 pictures"


def test_list_files_with_an_empty_corpus(monkeypatch):
    monkeypatch.setattr(list_files, "get_es_client", lambda: SimpleNamespace(search=lambda index, **kwargs: {}))

    assert tool("list_files", context()).invoke({}) == "No files have been ingested."
