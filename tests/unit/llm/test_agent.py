import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from search_engine.core.exceptions import LLMUnavailableError
from search_engine.llm.agent import LLMAgent, cited_numbers, format_context, format_history

CHUNKS = [
    Document("|Region|Revenue|\n|North|120|", metadata={"file_name": "report.pdf", "page": 3, "content": "table",
                                                       "modality": "text"}),
    Document("Revenue grew by twelve percent.", metadata={"file_name": "review.mp4", "timestamp": 75.4,
                                                           "content": "text", "modality": "video_transcript"}),
    Document("chart from report.pdf, page 4", metadata={"file_name": "report.pdf", "page": 4, "content": "chart",
                                                         "modality": "image"}),
]


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


def test_answer_sends_question_context_and_history_to_the_llm():
    llm = GenericFakeChatModel(messages=iter([AIMessage("North had 120 [1].")]))
    seen = []
    llm_invoke = llm.invoke
    object.__setattr__(llm, "invoke", lambda prompt: seen.append(prompt.to_string()) or llm_invoke(prompt))

    answer = LLMAgent(llm=llm).answer("Which region leads?", CHUNKS, [HumanMessage("earlier question")])

    assert answer == "North had 120 [1]."
    assert "Question: Which region leads?" in seen[0] and "[1] (report.pdf, page 3, table)" in seen[0]
    assert "User: earlier question" in seen[0] and "Cite every claim" in seen[0]


def test_gpt_oss_style_citations_become_square_brackets():
    llm = GenericFakeChatModel(messages=iter([AIMessage("North leads 【1】, the chart 【3†L1-L2】 and 【1, 2】.")]))

    answer = LLMAgent(llm=llm).answer("q", CHUNKS)

    assert answer == "North leads [1], the chart [3] and [1, 2]."
    assert cited_numbers(answer, count=3) == [1, 3, 2]


def test_llm_errors_and_empty_answers_raise_llm_unavailable():
    class Failing:
        def invoke(self, prompt):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(LLMUnavailableError, match="429"):
        LLMAgent(llm=Failing()).answer("q", CHUNKS)
    with pytest.raises(LLMUnavailableError, match="empty"):
        LLMAgent(llm=GenericFakeChatModel(messages=iter([AIMessage("  ")]))).answer("q", CHUNKS)
