import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from search_engine.core.config import Settings
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.llm.agent import LLMAgent, cited_numbers, format_history, normalise_citations
from search_engine.llm.context import format_context
from tests.unit.llm.fakes import ScriptedModel, tool_call

CHUNKS = [
    Document("|Region|Revenue|\n|North|120|", metadata={"chunk_id": "r-0", "file_name": "report.pdf", "page": 3,
                                                       "content": "table", "modality": "text"}),
    Document("Revenue grew by twelve percent.", metadata={"chunk_id": "v-0", "file_name": "review.mp4",
                                                           "timestamp": 75.4, "content": "text",
                                                           "modality": "video_transcript"}),
    Document("chart from report.pdf, page 4", metadata={"chunk_id": "r-1", "file_name": "report.pdf", "page": 4,
                                                         "content": "chart", "modality": "image"}),
]


class FakeReranker:
    def __init__(self, docs):
        self.docs, self.calls = docs, []

    def retrieve(self, query, filters, top_k):
        self.calls.append((query, top_k))
        return self.docs


def agent(*models, reranker=None, mcp_tools=(), **settings):
    return LLMAgent(Settings(**settings), models=list(models), reranker=reranker, mcp_tools=list(mcp_tools))


def prompt_of(model, call=0) -> str:
    return "\n".join(str(message.content) for message in model.seen[call])


def test_context_numbers_each_chunk_with_where_it_comes_from():
    context = format_context(CHUNKS)

    assert context.startswith("[1] (report.pdf, page 3, table)\n|Region|Revenue|")
    assert "[2] (review.mp4, at 01:15, text)\nRevenue grew" in context
    assert "[3] (report.pdf, page 4, chart) A picture; only this description is known: chart from report.pdf" in context


def test_history_is_rendered_as_a_short_transcript():
    assert format_history([]) == "(none)"
    assert format_history([HumanMessage("q"), AIMessage("a" * 900)]) == "User: q\nAssistant: " + "a" * 500


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("North leads [1].", [1]),
        ("Both [2][1], see also [1, 3].", [2, 1, 3]),
        ("Out of range [0] [4] [99] and [2].", [2]),
        ("No citations here.", []),
    ],
)
def test_cited_numbers(answer, expected):
    assert cited_numbers(answer, count=3) == expected


def test_gpt_oss_style_citations_become_square_brackets_and_tool_marks_are_dropped():
    answer = normalise_citations("North leads 【1】, the chart 【3†L1-L2】 and 【1, 2】; 32.8 % of the total[1]【calculator】.")

    assert answer == "North leads [1], the chart [3] and [1, 2]; 32.8 % of the total[1]."


def test_answers_from_the_context_without_tools_when_it_is_enough():
    model = ScriptedModel(replies=[AIMessage("North had 120 【1】.")])

    reply = agent(model).answer("Which region leads?", CHUNKS, [HumanMessage("earlier question")])

    assert reply.text == "North had 120 [1]." and reply.tools_used == [] and reply.chunks == CHUNKS
    prompt = prompt_of(model)
    assert "Question: Which region leads?" in prompt and "[1] (report.pdf, page 3, table)" in prompt
    assert "User: earlier question" in prompt and "Cite every claim" in prompt
    assert model.tool_names == ["search_documents", "read_surrounding", "list_files", "calculator"]


def test_calls_a_tool_and_answers_with_its_result():
    model = ScriptedModel(replies=[tool_call("calculator", expression="120 + 95"), AIMessage("Together 215 [1].")])

    reply = agent(model).answer("North plus South?", CHUNKS)

    assert reply.text == "Together 215 [1]." and reply.tools_used == ["calculator"]
    assert reply.tool_results == ["calculator: 215"]  # kept as evidence for EVAL
    results = [m.content for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert results == ["215"]  # the tool ran and its result went back to the LLM


def test_passages_a_tool_finds_continue_the_numbering_and_can_be_cited():
    found = Document("South had 95.", metadata={"chunk_id": "r-7", "file_name": "report.pdf", "page": 5,
                                                 "content": "text", "modality": "text"})
    reranker = FakeReranker([CHUNKS[0], found])  # one already in the context, one new
    model = ScriptedModel(replies=[tool_call("search_documents", query="South revenue"), AIMessage("South had 95 [4].")])

    reply = agent(model, reranker=reranker, agent_tool_top_k=5).answer("And South?", CHUNKS)

    assert reranker.calls == [("South revenue", 5)]
    assert [d.metadata["chunk_id"] for d in reply.chunks] == ["r-0", "v-0", "r-1", "r-7"]
    [result] = [m.content for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert result == "[4] (report.pdf, page 5, text)\nSouth had 95."
    assert reply.text == "South had 95 [4]." and reply.tools_used == ["search_documents"]
    assert reply.tool_results == []  # found passages are in the numbered context already


def test_tool_calls_are_capped():
    calls = [tool_call("calculator", f"call-{i}", expression=f"{i} + 1") for i in range(3)]
    model = ScriptedModel(replies=[*calls, AIMessage("Done [1].")])

    reply = agent(model, agent_max_tool_calls=2).answer("q", CHUNKS)

    results = [str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert results[:2] == ["1", "2"]  # two calls ran
    assert "limit" in results[2].lower()  # the third was refused, and the LLM was told to answer
    assert reply.text == "Done [1]."


def test_an_mcp_tool_is_offered_and_called_like_a_local_tool():
    async def lookup(term: str) -> str:
        return f"wiki says {term} is a colour"

    mcp_tool = StructuredTool.from_function(coroutine=lookup, name="wiki_lookup", description="Look up a term.")
    model = ScriptedModel(replies=[tool_call("wiki_lookup", term="teal"), AIMessage("Teal is a colour.")])

    reply = agent(model, mcp_tools=[mcp_tool]).answer("What is teal?", CHUNKS)

    assert model.tool_names[-1] == "wiki_lookup" and reply.tools_used == ["wiki_lookup"]
    assert [m.content for m in model.seen[1] if isinstance(m, ToolMessage)] == ["wiki says teal is a colour"]


def test_the_other_provider_answers_when_the_first_fails():
    failing = ScriptedModel(replies=[RuntimeError("429 rate limit")])
    backup = ScriptedModel(replies=[AIMessage("North had 120 [1].")])

    reply = agent(failing, backup).answer("q", CHUNKS)

    assert reply.text == "North had 120 [1]." and len(backup.seen) == 1


def test_a_failed_agent_run_falls_back_to_one_direct_answer():
    # the agent run ends without text (an empty reply); the direct call then answers from the same context
    model = ScriptedModel(replies=[AIMessage(""), AIMessage("North had 120 [1].")])

    reply = agent(model).answer("q", CHUNKS)

    assert reply.text == "North had 120 [1]." and len(model.seen) == 2


def test_when_no_llm_can_answer_it_is_llm_unavailable():
    broken = ScriptedModel(replies=[RuntimeError("503"), RuntimeError("503 again")])

    with pytest.raises(LLMUnavailableError, match="503"):
        agent(broken).answer("q", CHUNKS)
